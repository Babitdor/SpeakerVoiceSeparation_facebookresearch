# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import json
import logging
from pathlib import Path
import tqdm
import time
import gc
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from . import distrib
from .separate import separate
from .evaluate import evaluate
from .models.sisnr_loss import cal_loss
from .scripts.utils import (
    bold,
    copy_state,
    pull_metric,
    serialize_model,
    swap_state,
    LogProgress,
)


logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, model, optimizer, args, scaler, writer=None):
        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.tt_loader = data["tt_loader"]
        self.model = model
        self.writer = writer
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        self.scaler = scaler
        self.args = args

        self._init_components(args)
        self._reset()

        # Track memory usage
        self.memory_stats = {"peak_allocated": 0, "peak_cached": 0}

    def _init_components(self, args):

        # Learning schedular
        if args.lr_sched == "step":
            self.sched = StepLR(
                self.optimizer, step_size=args.step.step_size, gamma=args.step.gamma
            )
        elif args.lr_sched == "plateau":
            self.sched = ReduceLROnPlateau(
                self.optimizer,
                factor=args.plateau.factor,
                patience=args.plateau.patience,
            )
        else:
            self.sched = None

        # Training config
        self.device = args.device
        self.epochs = args.epochs
        self.max_norm = args.max_norm
        self.current_epoch = 0

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = Path(args.checkpoint_file) if args.checkpoint else None
        if self.checkpoint:
            logger.debug("Checkpoint will be saved to %s", self.checkpoint.resolve())
        self.history_file = args.history_file
        self.best_state = None
        self.restart = args.restart
        # keep track of losses
        self.history = []
        # Where to save samples
        self.samples_dir = args.samples_dir
        # logging
        self.num_prints = args.num_prints

    def _serialize(self, path):
        package = {
            "model": serialize_model(self.model),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
            "best_state": self.best_state,
            "args": self.args,
            "current_epoch": self.current_epoch,
        }
        torch.save(package, path)
        del package
        gc.collect()

    def _update_memory_state(self):
        """Track peak memory usage"""
        self.memory_stats["peak_allocated"] = max(
            self.memory_stats["peak_allocated"],
            torch.cuda.max_memory_allocated() / (1024**2),
        )
        self.memory_stats["peak_cached"] = max(
            self.memory_stats["peak_cached"],
            torch.cuda.max_memory_reserved() / (1024**2),
        )
        torch.cuda.reset_peak_memory_stats()

    def _log_memory(self, epoch):
        """Log memory usage to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(
                "Memory/Peak_Allocated_MB", self.memory_stats["peak_allocated"], epoch
            )
            self.writer.add_scalar(
                "Memory/Peak_Cached_MB", self.memory_stats["peak_cached"], epoch
            )
            logger.info(
                f"Memory Usage - Allocated: {self.memory_stats['peak_allocated']:.2f}MB, "
                f"Cached: {self.memory_stats['peak_cached']:.2f}MB"
            )

    def _cleanup(self):
        """Explicit memory cleanup"""
        torch.cuda.empty_cache()
        gc.collect()

    def _reset(self):
        load_from = None
        # Reset
        if self.checkpoint and self.checkpoint.exists() and not self.restart:
            load_from = self.checkpoint
        elif self.continue_from:
            load_from = self.continue_from

        if load_from:
            logger.info(f"Loading checkpoint model: {load_from}")
            # Load with minimal memory overhead
            package = torch.load(load_from, map_location="cpu", weights_only=False)
            if load_from == self.continue_from and self.args.continue_best:
                self.model.load_state_dict(package["best_state"])
            else:
                self.model.load_state_dict(package["model"]["state"])

            if "optimizer" in package and not self.args.continue_best:
                self.optimizer.load_state_dict(package["optimizer"])
            self.history = package["history"]
            self.best_state = package["best_state"]
            self.current_epoch = package.get("current_epoch", 0)

            del package
            gc.collect()

    def train(self):
        try:
            self._train_loop()
        finally:
            if self.writer is not None:
                self.writer.close()
            self._cleanup()

    def _train_loop(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                self._log_metrics(metrics, epoch)
                info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
                logger.info(f"Epoch {epoch}: {info}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self._train_epoch(epoch)
            self._validate_epoch(epoch)
            self._evaluate_if_needed(epoch)
            self._save_checkpoint()
            self._update_memory_state()
            self._log_memory(epoch)
            self._cleanup()

    def _train_epoch(self, epoch):
        # for epoch in range(len(self.history), self.epochs):

        # Train one epoch
        self.model.train()  # Turn on BatchNorm & Dropout
        logger.info("-" * 70)
        logger.info("Training...")

        start_time = time.time()
        train_loss = self._run_one_epoch(epoch)
        train_time = time.time() - start_time
        # Logging Training Loss (Terminal and tensorboard)
        logger.info(
            bold(
                f"Train Summary | End of Epoch {epoch + 1} | "
                f"Time {train_time:.2f}s | Loss {train_loss:.5f}"
            )
        )
        metrics = {"train": train_loss}
        self._log_metrics(metrics, epoch)
        # if self.writer is not None:
        #     self.writer.add_scalar("Loss/train", train_loss, epoch)

    def _validate_epoch(self, epoch):
        # Cross validation
        logger.info("-" * 70)
        logger.info("Cross validation...")
        self.model.eval()  # Turn off Batchnorm & Dropout

        start_time = time.time()
        with torch.no_grad():
            valid_loss = self._run_one_epoch(epoch, cross_valid=True)
        valid_time = time.time() - start_time
        # Logging Validation Loss (Terminal and tensorboard)
        logger.info(
            bold(
                f"Valid Summary | Epoch {epoch + 1} | "
                f"Time {valid_time:.2f}s | Loss {valid_loss:.5f}"
            )
        )

        # if self.writer is not None:
        #     self.writer.add_scalar("Loss/Validation", valid_loss, epoch)

        # learning rate scheduling
        if self.sched:
            if self.args.lr_sched == "plateau":
                self.sched.step(valid_loss)
            else:
                self.sched.step()

            # Logging the learning rate
            logger.info(
                f'Learning rate : {self.optimizer.state_dict()["param_groups"][0]["lr"]:.5f}'
            )
            if self.writer is not None:
                self.writer.add_scalar(
                    "Learning Rate",
                    self.optimizer.state_dict()["param_groups"][0]["lr"],
                    epoch,
                )

        # Logging and saving Best loss model
        best_loss = min(pull_metric(self.history, "valid") + [valid_loss])
        metrics = {"valid": valid_loss, "best": best_loss}
        # Save the best model
        if valid_loss == best_loss or self.args.keep_last:
            logger.info(bold("New best valid loss %.4f"), valid_loss)
            self.best_state = copy_state(self.model.state_dict())

        self._log_metrics(metrics, epoch)
        self.history.append({**metrics, "train": metrics.get("train", 0)})

        # evaluate and separate samples every 'eval_every' argument number of epochs
        # also evaluate on last epoch

    def _evaluate_if_needed(self, epoch):
        if (epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1:
            # Evaluate on the testset
            logger.info("-" * 70)
            logger.info("Evaluating on the test set...")
            # We switch to the best known model for testing
            with swap_state(self.model, self.best_state), torch.no_grad():
                sisnr, pesq, stoi = evaluate(
                    self.args, self.model, self.tt_loader, self.args.sample_rate
                )
            metrics = {"sisnr": sisnr, "pesq": pesq, "stoi": stoi}
            self._log_metrics(metrics, epoch)
            self.history[-1].update(metrics)
            # info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            # logger.info("-" * 70)
            # logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            # separate some samples
            logger.info("Separate and save samples...")
            separate(self.args, self.model, self.samples_dir)
            self._cleanup()

        # info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
        # logger.info("-" * 70)
        # logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

    def _save_checkpoint(self):
        """Save checkpoint if needed"""
        if distrib.rank == 0:
            json.dump(self.history, open(self.history_file, "w"), indent=2)
            # Save model each epoch
            if self.checkpoint:
                self._serialize(self.checkpoint)
                logger.debug("Checkpoint saved to %s", self.checkpoint.resolve())

    def _log_metrics(self, metrics, epoch):
        """Log metrics to both console and TensorBoard"""
        info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
        logger.info(bold(f"Epoch {epoch + 1} | {info}"))

        if self.writer is not None:
            for metric, value in metrics.items():
                if metric in ["train", "valid", "best"]:
                    self.writer.add_scalar(f"Loss/{metric}", value, epoch)
                elif metric in ["sisnr", "pesq", "stoi"]:
                    self.writer.add_scalar(f"Metrics/{metric}", value, epoch)

            self.writer.add_scalar(
                "LearningRate", self.optimizer.param_groups[0]["lr"], epoch
            )

    def _run_one_epoch(self, epoch, cross_valid=False):
        """Memory-optimized epoch runner"""
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        data_loader.epoch = epoch

        label = "Train" if not cross_valid else "Valid"
        logprog = LogProgress(
            logger,
            data_loader,
            updates=self.num_prints,
            name=f"{label} | Epoch {epoch + 1}",
        )
        desc = f"{label} | Epoch {epoch + 1}"
        with tqdm.tqdm(data_loader, desc=desc, unit="batch") as pbar:
            for i, data in enumerate(pbar):
                # Process data with memory efficiency
                mixture, lengths, sources = [
                    x.to(self.device, non_blocking=True) for x in data
                ]

                # Forward pass with memory management
                with torch.amp.autocast(
                    device_type="cuda", enabled=self.scaler is not None
                ), torch.set_grad_enabled(not cross_valid):

                    estimate_source = self.dmodel(mixture)
                    if cross_valid:
                        estimate_source = estimate_source[-1:]

                    loss = self._compute_loss(estimate_source, sources, lengths)

                # Backward pass if training
                if not cross_valid:
                    self._optimize_step(loss)

                # Update progress and clean up
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{total_loss / (i + 1):.5f}")
                logprog.update(loss=format(total_loss / (i + 1), ".5f"))

                # Clean up to save memory
                del mixture, lengths, sources, estimate_source, loss
                if i % 10 == 0:
                    self._cleanup()

        return distrib.average([total_loss / (i + 1)], i + 1)[0]

    def _compute_loss(self, estimate_source, sources, lengths):
        """Compute loss with memory efficiency"""
        loss = 0
        cnt = len(estimate_source)

        for c_idx, est_src in enumerate(estimate_source):
            coeff = (c_idx + 1) * (1 / cnt)
            sisnr_loss, _, _, _ = cal_loss(sources, est_src, lengths)
            loss += coeff * sisnr_loss

        return loss / len(estimate_source)

    def _optimize_step(self, loss):
        """Perform optimization step with memory management"""
        self.optimizer.zero_grad()  # More memory efficient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()
