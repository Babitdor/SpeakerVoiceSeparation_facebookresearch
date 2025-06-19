from pathlib import Path
import glob
import pandas as pd
import soundfile as sf
import numpy as np
import os
from scipy.signal import resample_poly
from tqdm import tqdm
import argparse

FS_ORIG = 16000

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--wsj0_path", default="../")
parser.add_argument("-o", "--output_folder", default="wsj0-mix")
parser.add_argument("-n", "--n_src", default=2, type=int)
parser.add_argument("-sr", "--samplerate", default=8000, type=int)
parser.add_argument("--len_mode", nargs="+", type=str, default=["min", "max"])
args = parser.parse_args()


# Read activlev file. Build {utt_id: activlev} dict
current_path = os.path.join(os.getcwd(), "pywsj0_mix")

txt_files = glob.glob(os.path.join(current_path, "metadata", "activlev", "*.txt"))
print(f"Found {len(txt_files)} activlev files")
if not txt_files:
    raise FileNotFoundError("No .txt files found in 'metadata' folder")

activlev_df = pd.concat(
    [
        pd.read_csv(
            txt_f, delimiter=" ", names=["utt", "alev"], index_col=False, header=None
        )
        for txt_f in txt_files
    ]
)

activlev_dic = dict(zip(activlev_df.utt, activlev_df.alev))


def wavwrite_quantize(samples):
    return np.int16(np.round((2**15) * samples))


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype="PCM_16")


for cond in ["tr", "cv", "tt"]:
    # Output folders (wav8k-16k/min-max/tr-cv-tt/mix-src{i})
    base = (
        Path(args.output_folder)
        / f"{args.n_src}speakers"
        / f"wav{args.samplerate // 1000}k"
    )
    min_mix_folder = base / "min" / cond / "mix"
    min_src_folders = [base / "min" / cond / f"s{i+1}" for i in range(args.n_src)]
    max_mix_folder = base / "max" / cond / "mix"
    max_src_folders = [base / "max" / cond / f"s{i+1}" for i in range(args.n_src)]
    for p in min_src_folders + max_src_folders + [min_mix_folder, max_mix_folder]:
        p.mkdir(parents=True, exist_ok=True)

    # Read SNR scales file
    header = [
        x
        for t in zip(
            [f"s_{i}" for i in range(args.n_src)],
            [f"snr_{i}" for i in range(args.n_src)],
        )
        for x in t
    ]
    mix_file_path = os.path.join(
        current_path, "metadata", f"mix_{args.n_src}_spk_{cond}.txt"
    )

    mix_df = pd.read_csv(
        mix_file_path,
        delimiter=" ",
        names=header,
        index_col=False,
    )

    # print(args.wsj0_path)
    for idx in tqdm(range(len(mix_df))):
        sources = []
        skip_example = False  # Flag to skip this mixture

        for i in range(args.n_src):
            relative_path = mix_df[f"s_{i}"][idx]
            full_path = os.path.join(args.wsj0_path, relative_path).replace("/", os.sep)

            if not os.path.isfile(full_path):
                # print(
                #     f"[WARNING] Missing file: {full_path}, skipping example index {idx}"
                # )
                skip_example = True
                break  # stop this loop and skip the whole example

            audio, _ = sf.read(full_path, dtype="float32")
            sources.append(audio)

        if skip_example:
            continue

        snrs = [mix_df[f"snr_{i}"][idx] for i in range(args.n_src)]
        # print(f"Loaded sources: {[s.shape for s in sources]}")

        resampled_sources = [
            resample_poly(s, args.samplerate, FS_ORIG) for s in sources
        ]
        # print(f"Resampled sources: {[s.shape for s in resampled_sources]}")
        min_len, max_len = min([len(s) for s in resampled_sources]), max(
            [len(s) for s in resampled_sources]
        )
        padded_sources = [
            np.hstack((s, np.zeros(max_len - len(s)))) for s in resampled_sources
        ]

        activlev_scales = [
            activlev_dic[mix_df[f"s_{i}"][idx].split("/")[-1].replace(".wav", "")]
            for i in range(args.n_src)
        ]
        # activlev_scales = [np.sqrt(np.mean(s**2)) for s in resampled_sources]  # If no activlev file
        scaled_sources = [
            s / np.sqrt(scale) * 10 ** (x / 20)
            for s, scale, x in zip(padded_sources, activlev_scales, snrs)
        ]

        sources_np = np.stack(scaled_sources, axis=0)
        mix_np = np.sum(sources_np, axis=0)

        # Merge filenames for mixture name.  (when mixing weight is 0.450124, it truncates 0.45012, hence the 10x)
        matlab_round = lambda x, y: round(x, y) if abs(x) >= 1.0 else round(x, y + 1)
        pp = lambda x: (
            x.split("/")[-1].replace(".wav", "")
            if isinstance(x, str)
            else "{:12.8g}".format(x).strip()
        )
        filename = "_".join([pp(mix_df[u][idx]) for u in header]) + ".wav"

        if "max" in args.len_mode:
            gain = (
                np.max([1.0, np.max(np.abs(mix_np)), np.max(np.abs(sources_np))]) / 0.9
            )
            mix_np_max = mix_np / gain
            sources_np_max = sources_np / gain
            wavwrite(max_mix_folder / filename, mix_np_max, args.samplerate)
            for s_fold, src_np in zip(max_src_folders, sources_np_max):
                wavwrite(s_fold / filename, src_np, args.samplerate)
        if "min" in args.len_mode:
            sources_np = sources_np[:, :min_len]
            mix_np = mix_np[:min_len]
            gain = (
                np.max([1.0, np.max(np.abs(mix_np)), np.max(np.abs(sources_np))]) / 0.9
            )
            mix_np /= gain
            sources_np /= gain
            wavwrite(min_mix_folder / filename, mix_np[:min_len], args.samplerate)
            for s_fold, src_np in zip(min_src_folders, sources_np):
                wavwrite(s_fold / filename, src_np[:min_len], args.samplerate)
