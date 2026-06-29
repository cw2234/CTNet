"""
We would like to thank Xiaolu Jiang for her contribution to data preprocessing.
Author : Xiaolu Jiang
modified by Wei Zhao

Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊

Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7


"""

import mne
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import scipy.io as sio
import argparse
import os

BCICIV_2b_gdf_DIR_NAME = "BCICIV_2b_gdf"
TRUE_LABEL_DIR_NAME = "true_labels"


def changeTrainGdf2Mat(dir_path, out_dir):
    # 读取训练集和对应的标签到mat文件
    for nSub in range(1, 10):
        data_sub = np.empty((0, 3, 1000))
        labels_sub = np.empty((0, 1))
        for nSes in range(1, 4):
            # Load the gdf file
            data_file_path = os.path.join(
                dir_path, BCICIV_2b_gdf_DIR_NAME, f"B0{nSub}0{nSes}T.gdf"
            )
            raw = mne.io.read_raw_gdf(data_file_path)
            # Select the events of interest
            # Events is the data at each time point, and event_dict is the correspondence between the label and the label sequence number.
            events, event_dict = mne.events_from_annotations(raw)
            event_id = {"Left": event_dict["769"], "Right": event_dict["770"]}
            # Select the events corresponding to the four categories we are concerned about. Here events[:, 2] refers to the third column in events, that is, the event number.
            selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

            # Select the removed channel, that is, the EOG channel does not participate in classification
            raw.info["bads"] += ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
            picks = mne.pick_types(
                raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
            )
            # Epoch the data
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

            # Get the labels
            # labels = epochs.events[:, 2]
            label_file_path = os.path.join(
                dir_path, TRUE_LABEL_DIR_NAME, f"B0{nSub}0{nSes}T.mat"
            )
            mat = sio.loadmat(label_file_path)
            labels = mat["classlabel"]

            # Store the data and labels of each epoch into data_sub and labels_sub
            data_sub = np.vstack((data_sub, epochs.get_data()))
            labels_sub = np.vstack((labels_sub, labels))

        # Output the shape of data_sub and labels_sub to ensure that it is consistent with the data in the mat file required by conformer.py
        print("B0%dT:" % nSub, data_sub.shape, labels_sub.shape)
        # Save the data and labels to a .mat file
        result_file_path = os.path.join(out_dir, f"B0{nSub}T.mat")
        savemat(
            result_file_path,
            {"data": data_sub, "label": labels_sub},
        )


def changeTestGdf2Mat(dir_path, out_dir):
    # Read the test set and the corresponding labels into a mat file
    for nSub in range(1, 10):
        data_sub = np.empty((0, 3, 1000))
        labels_sub = np.empty((0, 1))
        for nSes in range(4, 6):
            # Load the gdf file
            data_file_path = os.path.join(
                dir_path, BCICIV_2b_gdf_DIR_NAME, f"B0{nSub}0{nSes}E.gdf"
            )
            raw = mne.io.read_raw_gdf(data_file_path)

            # Select the events of interest
            events, event_dict = mne.events_from_annotations(raw)
            event_id = {"Unknown": event_dict["783"]}
            selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

            raw.info["bads"] += ["EOG:ch01", "EOG:ch02", "EOG:ch03"]
            picks = mne.pick_types(
                raw.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
            )

            # Epoch the data
            epochs = mne.Epochs(
                raw,
                selected_events,
                event_id,
                picks=picks,
                tmin=0,
                tmax=3.996,
                preload=True,
                baseline=None,
                on_missing="ignore",
            )

            # Get the labels
            label_file_path = os.path.join(
                dir_path, TRUE_LABEL_DIR_NAME, f"B0{nSub}0{nSes}E.mat"
            )
            mat = sio.loadmat(label_file_path)
            labels = mat["classlabel"]

            data_sub = np.vstack((data_sub, epochs.get_data()))
            labels_sub = np.vstack((labels_sub, labels))

        print("B0%dE:" % nSub, data_sub.shape, labels_sub.shape)
        # Save the data and labels to a .mat file
        result_file_path = os.path.join(out_dir, f"B0{nSub}E.mat")
        savemat(
            result_file_path,
            {"data": data_sub, "label": labels_sub},
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GDF files to mat files.",
        usage="python preprocessing_for_2b.py --dir_path ./data/BCICIV_2b --out_dir ./data/BCICIV_2b_mat_raw",
    )
    parser.add_argument(
        "--dir_path", type=str, default="./data/BCICIV_2b", help="GDF file dir path."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data/BCICIV_2b_mat_raw",
        help="output directory.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Create directory {args.out_dir}")

    changeTrainGdf2Mat(args.dir_path, args.out_dir)
    changeTestGdf2Mat(args.dir_path, args.out_dir)


if __name__ == "__main__":
    main()
