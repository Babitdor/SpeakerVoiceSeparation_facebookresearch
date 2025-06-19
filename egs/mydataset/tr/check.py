import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_alignment(mix, s1, s2):
    len_mix = len(mix)
    len_s1 = len(s1)
    len_s2 = len(s2)
    print(f"Lengths: mix={len_mix}, s1={len_s1}, s2={len_s2}")

    min_len = min(len_mix, len_s1, len_s2)
    max_len = max(len_mix, len_s1, len_s2)

    if len_mix != len_s1 or len_mix != len_s2:
        print("WARNING: Lengths are not equal!")
        print(f"mix: {len_mix}, s1: {len_s1}, s2: {len_s2}")

    # Check for missing indices
    for name, arr, l in [("mix", mix, len_mix), ("s1", s1, len_s1), ("s2", s2, len_s2)]:
        if l < max_len:
            print(f"{name} is missing {max_len - l} entries at the end.")

    # Check for misalignment (here, just checks for presence, not value equality)
    for i in range(min_len):
        if mix[i] is None or s1[i] is None or s2[i] is None:
            print(f"Missing entry at index {i}: mix={mix[i]}, s1={s1[i]}, s2={s2[i]}")

    # Optionally, check for value equality (if that's what you want)
    # for i in range(min_len):
    #     if mix[i] != s1[i] or mix[i] != s2[i]:
    #         print(f"Misaligned at index {i}: mix={mix[i]}, s1={s1[i]}, s2={s2[i]}")

    print("Check complete.")


if __name__ == "__main__":
    # Replace with your actual file paths
    mix = load_json("mix.json")
    s1 = load_json("s1.json")
    s2 = load_json("s2.json")

    check_alignment(mix, s1, s2)
