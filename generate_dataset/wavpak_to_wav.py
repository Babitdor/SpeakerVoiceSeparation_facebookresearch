import os
from pathlib import Path
from tqdm import tqdm  # <-- Import tqdm

# Set the WSJ0 root directory (your actual dataset path)
root_dir = Path(
    r"D:\Personal Projects\Speech\SVoice\svoice\dataset\csr_1_LDC93S6A\csr_1"
)

# List all disc directories, skipping "readme.txt", "11-13.1", and "file.tbl"
disc_dir = []
for list_disc in os.listdir(root_dir):
    if list_disc not in ["readme.txt", "11-13.1", "file.tbl"]:
        disc_dir.append(root_dir / list_disc / "wsj0")
        print(list_disc)

# Define target directory for output
my_path = Path(
    r"D:\Personal Projects\Speech\SVoice\svoice\dataset\csr_1_LDC93S6A\csr_1"
)
my_path.mkdir(parents=True, exist_ok=True)

# Path to sph2pipe executable
sph2pipe_path = Path(r".\sph2pipe\sph2pipe.exe")  # Adjust if needed

# Iterate through each disc
for list_sub_data in tqdm(disc_dir, desc="Discs"):
    for sub_data_dir in tqdm(os.listdir(list_sub_data), desc="Subdirs", leave=False):
        if not (sub_data_dir.startswith("si") or sub_data_dir.startswith("sd")):
            continue

        s_dir = my_path / sub_data_dir
        # s_dir.mkdir(exist_ok=True)

        datatype_dir = list_sub_data / sub_data_dir
        for list_spk in tqdm(os.listdir(datatype_dir), desc="Speakers", leave=False):
            # spk_dir = s_dir / list_spk
            spk_dir_abs = datatype_dir / list_spk
            # spk_dir.mkdir(exist_ok=True)

            for wv_file in tqdm(os.listdir(spk_dir_abs), desc="Files", leave=False):
                if not (wv_file.endswith(".wv1") or wv_file.endswith(".wv2")):
                    continue

                speech_path = spk_dir_abs / wv_file
                if wv_file.endswith(".wv1"):
                    target_name = speech_path.stem + ".wav"
                else:
                    target_name = speech_path.stem + "_1.wav"
                target_path = spk_dir_abs / target_name

                # Construct and run sph2pipe command
                cmd = f"powershell -Command \"& '{str(sph2pipe_path)}' -f wav '{str(speech_path)}' '{str(target_path)}'\""
                os.system(cmd)
