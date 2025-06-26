# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Authors: Yossi Adi (adiyoss) and Alexandre Défossez (adefossez)

import json
import logging
import math
from pathlib import Path
import os
import re

import librosa
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

from .preprocess import preprocess_one_dir
from .audio import Audioset

logger = logging.getLogger(__name__)


def sort(infos):
    return sorted(infos, key=lambda info: int(info[1]), reverse=True)


class Trainset:
    def __init__(
        self,
        json_dir,
        sample_rate=16000,
        segment=4.0,
        stride=1.0,
        pad=True,
        subset=False,
        subset_size=500,
    ):
        # This is safe
        mix_json = os.path.join(json_dir, "mix.json")
        s_jsons = list()
        s_infos = list()
        sets_re = re.compile(r"s[0-9].json")
        for s in os.listdir(json_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(json_dir, s))

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, "r") as f:
                s_infos.append(json.load(f))

        if subset and subset_size is not None:
            mix_infos = mix_infos[:subset_size]
            s_infos = [s[:subset_size] for s in s_infos]

        length = int(sample_rate * segment)
        stride = int(sample_rate * stride)

        kw = {"length": length, "stride": stride, "pad": pad}
        self.mix_set = Audioset(sort(mix_infos), **kw)

        self.sets = list()
        for s_info in s_infos:
            self.sets.append(Audioset(sort(s_info), **kw))

        # verify all sets has the same size
        for idx, s in enumerate(self.sets):
            # print(f"Set {idx}: len(s)={len(s)}, len(mix_set)={len(self.mix_set)}")
            assert len(s) == len(self.mix_set)

    def __getitem__(self, index):
        mix_sig = self.mix_set[index]
        tgt_sig = [self.sets[i][index] for i in range(len(self.sets))]
        return (
            self.mix_set[index],
            torch.LongTensor([mix_sig.shape[0]]),
            torch.stack(tgt_sig),
        )

    def __len__(self):
        return len(self.mix_set)


class Validset:
    """
    load entire wav.
    """

    def __init__(
        self,
        json_dir,
        sample_rate=16000,
        segment=None,
        pad=False,
        stride=None,
        subset=False,
        subset_size=500,
    ):
        mix_json = os.path.join(json_dir, "mix.json")
        s_jsons = list()
        s_infos = list()
        sets_re = re.compile(r"s[0-9].json")
        for s in os.listdir(json_dir):
            if sets_re.search(s):
                s_jsons.append(os.path.join(json_dir, s))
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        for s_json in s_jsons:
            with open(s_json, "r") as f:
                s_infos.append(json.load(f))

        if subset and subset_size is not None:
            mix_infos = mix_infos[:subset_size]
            s_infos = [s[:subset_size] for s in s_infos]

        kw = {}
        if segment is not None:
            kw["length"] = int(sample_rate * segment)
            kw["stride"] = int(sample_rate * stride)
            kw["pad"] = pad
        self.mix_set = Audioset(sort(mix_infos), **kw)
        self.sets = list()
        for s_info in s_infos:
            self.sets.append(Audioset(sort(s_info), **kw))
        for s in self.sets:
            assert len(s) == len(self.mix_set)

    def __getitem__(self, index):
        mix_sig = self.mix_set[index]
        tgt_sig = [self.sets[i][index] for i in range(len(self.sets))]

        return (
            self.mix_set[index],
            torch.LongTensor([mix_sig.shape[0]]),
            torch.stack(tgt_sig),
        )

    def __len__(self):
        return len(self.mix_set)


# The following piece of code was adapted from https://github.com/kaituoxu/Conv-TasNet
# released under the MIT License.
# Author: Kaituo XU
# Created on 2018/12
class EvalDataset(data.Dataset):
    """
    Yields file paths and sample rate, does NOT load audio in __init__ or __getitem__.
    """

    def __init__(
        self, mix_dir, mix_json, batch_size, sample_rate=8000, subset_size=100
    ):
        super(EvalDataset, self).__init__()
        assert mix_dir is not None or mix_json is not None
        if mix_dir is not None:
            preprocess_one_dir(mix_dir, mix_dir, "mix", sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, "mix.json")
        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        # sort by #samples (impl bucket)
        sorted_mix_infos = sorted(
            mix_infos, key=lambda info: int(info[1]), reverse=True
        )
        # minibatch is now a list of lists of file paths
        self.batches = [
            sorted_mix_infos[i : i + batch_size]
            for i in range(0, len(sorted_mix_infos), batch_size)
        ]
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        # Only return file paths and sample rate, not audio
        return self.batches[index], self.sample_rate

    def __len__(self):
        return len(self.batches)


class EvalDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, **kwargs):
        super(EvalDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(batch):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        filenames: a list contain B strings
    """
    # batch should be located in list
    assert len(batch) == 1
    batch_file_paths, sample_rate = batch[0]
    mixtures, filenames = [], []
    for mix_info in batch_file_paths:
        mix_path = mix_info[0]
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    ilens = np.array([mix.shape[0] for mix in mixtures])
    pad_value = 0
    mixtures_pad = pad_list(
        [torch.from_numpy(mix).float() for mix in mixtures], pad_value
    )
    ilens = torch.from_numpy(ilens)
    # Explicitly release memory
    del mixtures
    return mixtures_pad, ilens, filenames


def load_mixtures(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is T np.ndarray
        filenames: a list containing B strings
        T varies from item to item.
    """
    mixtures, filenames = [], []
    mix_infos, sample_rate = batch
    # for each utterance
    for mix_info in mix_infos:
        mix_path = mix_info[0]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        mixtures.append(mix)
        filenames.append(mix_path)
    return mixtures, filenames


def pad_list(xs, pad_value):
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]
    return pad
