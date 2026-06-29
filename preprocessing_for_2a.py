"""
# Function: Read files and preprocess them
# Steps:
# 1. Import data from the gdf file provided before the competition,
#    remove unwanted channels, and select required events.
# 2. Select desired time segments for slicing; treat each segment (4s) as one sample.
# 3. Import labels from the mat file provided after the competition,
#    ensuring they correspond with epochs and their numbers match.
# 4. Save the resulting data in a new mat file,
#    preparing it for use in the subsequent main.py.
# 每个数据集都放在 data 目录下，可以有 BCICIV_2a文件夹，目录结构如下：
# data/
# ├── BCICIV_2a/         # 存放数据集的目录
# │   ├── BCICIV_2a_gdf/
# │   ├── true_labels/
# │
# ├── BCICIV_2a_mat_raw/ # 存放处理后的mat文件


"""

import mne
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import argparse
import scipy.io as sio
import os


def changeGdf2Mat(dir_path, out_dir, mode="train"):
    """
    read data from GDF files and store as mat files

    Parameters
    ----------
    dir_path : str
        GDF file dir path, 这个目录下放了BCICIV_2a_gdf文件夹和true_labels文件夹.
    out_dir : str
        output directory, 这个目录下会放生成的mat文件.
    mode : str, optional
        change train dataset or eval dataset. The default is "train".

    Returns
    -------
    None.

    """
    mode_str = ""
    if mode == "train":
        mode_str = "T"
    else:
        mode_str = "E"
    for nSub in range(1, 10):
        # Load the gdf file
        data_file_path = os.path.join(
            dir_path, "BCICIV_2a_gdf", f"A0{nSub}{mode_str}.gdf"
        )
        raw = mne.io.read_raw_gdf(data_file_path)

        # Select the events of interest
        events, event_dict = mne.events_from_annotations(raw)
        if mode == "train":
            # train dataset are labeled
            event_id = {
                "Left": event_dict["769"],
                "Right": event_dict["770"],
                "Foot": event_dict["771"],
                "Tongue": event_dict["772"],
            }
        else:
            # evaluate dataset are labeled as 'Unknnow'
            event_id = {"Unknown": event_dict["783"]}

        # Select the events corresponding to the four categories we are interested in. Here, events[:, 2] refers to the third column of the events array, which represents the event IDs.
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

        # remove EOG channels
        raw.info["bads"] += ["EOG-left", "EOG-central", "EOG-right"]
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
        )
        # Epoch the data
        # using 4s (1000 sample point ) segmentation
        epochs = mne.Epochs(
            raw,
            selected_events,
            event_id,
            picks=picks,
            tmin=0,
            tmax=3.996,
            preload=True,
            baseline=None,
        )

        filtered_data = epochs.get_data()
        label_file_path = os.path.join(
            dir_path, "true_labels", f"A0{nSub}{mode_str}.mat"
        )
        mat = sio.loadmat(label_file_path)  # load target mat file
        labels = mat["classlabel"]

        # Save the data and labels to a .mat file
        result_file_path = os.path.join(out_dir, f"A0{nSub}{mode_str}.mat")
        savemat(result_file_path, {"data": filtered_data, "label": labels})


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GDF files to mat files.",
        usage="python preprocessing_for_2a.py --dir_path ./data/BCICIV_2a --out_dir ./data/BCICIV_2a_mat_raw",
    )
    parser.add_argument(
        "--dir_path", type=str, default="./data/BCICIV_2a", help="GDF file dir path."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/BCICIV_2a_mat_raw",
        help="output directory.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"create directory {args.out_dir}")
    # prepare train dataset
    changeGdf2Mat(args.dir_path, args.out_dir, "train")
    # prepare test dataset
    changeGdf2Mat(args.dir_path, args.out_dir, "eval")


if __name__ == "__main__":
    main()
