# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Authors: Yossi Adi (adiyoss) and  Alexandre Défossez (adefossez)

import json
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import socket
import subprocess as sp
import sys
import time

import hydra
from torch.amp import GradScaler
from svoice.executor import start_ddp_workers

logger = logging.getLogger(__name__)


def run(args):
    import torch
    from transvoice import distrib
    from transvoice.data.data import Trainset, Validset
    from transvoice.models.Transwave import TranSWave
    from transvoice.solver import Solver

    writer = None
    if hasattr(args.logging, "tensorboard") and args.logging.tensorboard.enable:
        writer = SummaryWriter(log_dir=args.logging.tensorboard.log_dir)

    logger.info("Running on host %s", socket.gethostname())
    distrib.init(args)

    if args.model == "swave":
        kwargs = dict(args.swave)
        kwargs["sr"] = args.sample_rate
        kwargs["segment"] = args.segment
        model = TranSWave(**kwargs)
    else:
        logger.fatal("Invalid model name %s", args.model)
        os._exit(1)

    # requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, "valid_length"):
        segment_len = int(args.segment * args.sample_rate)
        segment_len = model.valid_length(segment_len)
        args.segment = segment_len / args.sample_rate

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info("Size: %.1f MB", mb)
        if hasattr(model, "valid_length"):
            field = model.valid_length(1)
            logger.info("Field: %.1f ms", field / args.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    # Building datasets and loaders
    tr_dataset = Trainset(
        args.dset.train,
        sample_rate=args.sample_rate,
        segment=args.segment,
        stride=args.stride,
        pad=args.pad,
        subset=args.use_train_subset,
        subset_size=args.train_subset_size,
    )
    tr_loader = distrib.loader(
        tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # batch_size=1 -> use less GPU memory to do cv
    cv_dataset = Validset(
        args.dset.valid,
        segment=args.segment,
        stride=args.stride,
        pad=args.pad,
        subset=args.use_val_subset,
        subset_size=args.val_subset_size,
    )
    tt_dataset = Validset(
        args.dset.test,
        segment=args.segment,
        stride=args.stride,
        pad=args.pad,
        subset=args.use_val_subset,
        subset_size=args.val_subset_size,
    )
    cv_loader = distrib.loader(
        cv_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    tt_loader = distrib.loader(
        tt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model.cuda()
    elif torch.backends.mps.is_available():
        model.to("mps")
    else:
        model.cpu()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2)
        )
    else:
        logger.fatal("Invalid optimizer %s", args.optim)
        os._exit(1)

    scaler = GradScaler(device="cuda")

    # Construct Solver
    solver = Solver(data, model, optimizer, args, scaler=scaler, writer=writer)
    solver.train()


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    if args.ddp and args.rank is None:
        start_ddp_workers()
    else:
        run(args)


@hydra.main(config_path="conf", config_name="trans_config.yaml", version_base=None)
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
