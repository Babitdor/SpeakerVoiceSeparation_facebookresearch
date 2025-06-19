from pathlib import Path
import subprocess

dataset_path = "D:/Personal Projects/Speech/SVoice/svoice/dataset/csr_1_LDC93S6A/csr_1"
base_path = Path(dataset_path)
output_root = Path("D:/Personal Projects/Speech/SVoice/svoice/dataset/mix_splits_data")

for folder in base_path.iterdir():
    if folder.is_dir():
        audio_path = folder
        output_folder = output_root / folder.name
        output_folder.mkdir(parents=True, exist_ok=True)
        # print(audio_path)
        result = subprocess.run(
            [
                "python",
                "pywsj0_mix/generate_wsjmix.py",
                "-p",
                str(audio_path),
                "-o",
                str(output_folder),
                "-n",
                "2",
                "-sr",
                "8000",
                "--len_mode",
                "min",
                "max",
            ]
        )

        if result.returncode != 0:
            print(f"Error occurred when processing {folder}")
