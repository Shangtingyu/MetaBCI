# -*- coding: utf-8 -*-
# Organization: Hangzhou MindMatrixes Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/31
# License: MIT License

"""
This dataset is a non-public dataset.
if you need this dataset, please contact this mailbox:
xingjian.zhang@mindmatrixes.com
or download it here:
https://pan.baidu.com/s/1MwNy-EIkMlnUZhfKjlf5ig?pwd=mdsk
"""
import ntpath
import os
from pathlib import Path
import numpy as np
from metabci.brainda.datasets.sleep_apple import Sleep_Apples
from typing import Union, List, Dict, Optional


class Sleep_MSP(Sleep_Apples):
    """
    Methods:
        save_processed_data(subjects,select_ch,update_path):
            For the original dataset that has been stored (and will be downloaded automatically if it has not
            been downloaded yet), the original dataset is processed and saved as npz file in the specified path.
        get_processed_data(subjects,select_ch,update_path):
            Read the processed data file,return [labels, datas]
    """

    _EVENTS = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "R": 4
    }
    _CHANNELS = ["C3_M2", "ROC", "LOC"]

    def __init__(self, dataPath: str = None):
        self.dataPath = dataPath
        self.dataset_code = "msp"
        self.events = self._EVENTS
        self.channels = self._CHANNELS
        super().__init__(dataPath=dataPath)

    def save_processed_data(self,
                            subjects: List[Union[int, str]] = None,
                            path: Optional[Union[str, Path]] = None,
                            force_update: bool = False,
                            update_path: Optional[Union[str, Path]] = None,
                            proxies: Optional[Dict[str, str]] = None,
                            verbose: Optional[Union[bool, str, int]] = None,
                            select_ch: List[str] = None):
        """
        Save processed EEG/EOG data and corresponding labels as .npz files.

        Parameters:
        -----------
        subjects : list or None, optional
            List of subject indices to process. If None, processes all subjects.
        select_ch : list[str], optional
            Channel name(F4_M1, ROC or LOC) to select from the raw EEG data. If None, select_ch is ["EEG Fpz-Cz"]
        update_path : str or
            Path to save the processed data files. If None, defaults to a specified path.

        Notes:
        ------
        This function processes raw EEG data and its corresponding annotations (labels) and saves the segmented data into .npz files. It handles synchronization between
        raw data and annotations, filters out unwanted epochs, and ensures data integrity before saving.
        """
        if select_ch is None:
            select_ch = ["F4_M1"]
        savePath_ch = update_path
        raws = self.get_data(subjects)
        annotFiles = []
        for i in subjects:
            annotFiles.append(self.label_path(i))
        list_otherHZ = []
        for idx, subject in enumerate(subjects):
            rawdata = raws[subject]['session_0']['run_0']
            annotdata = self.readAnnotFiles(annotFiles[idx][0][0])
            print("annotdata: " + str(annotdata))
            sampling_rate = int(rawdata.info['sfreq'])
            if sampling_rate != 500:
                list_otherHZ.append(subject)
                continue
            raw_startTime = str(rawdata.info['meas_date']).split(" ")[1]
            raw_startTime = raw_startTime.split("+")[0]
            ann_startTime = annotdata.iloc[0, 1]

            # 检查数据和标签的时间是否对齐, 不对齐则矫正
            onsetSec, flag_del = self.checkTime(raw_startTime, ann_startTime)
            if flag_del:
                continue

            durationSecond = len(annotdata) * 30
            labels = annotdata.iloc[:, 0].to_numpy()
            mapping_function = np.frompyfunc(self._EVENTS.get, 1, 1)  # 将数组元素映射为对应的值
            labels = mapping_function(labels)
            data_idx = int(onsetSec * sampling_rate) + np.arange(durationSecond * sampling_rate, dtype=int)
            data_X, lable_y = self.cleanData(rawdata, labels, data_idx, select_ch, sampling_rate)
            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            print(data_X)
            save_dict = {
                "x": data_X,
                "y": lable_y,
                "fs": sampling_rate,
                "ch_label": select_ch,
            }
            np.savez(os.path.join(savePath_ch, filename), **save_dict)


if __name__ == "__main__":
    dataPath = r'D:\sleep-data\msp\edfs'
    path = r'D:\sleep-data\msp\F4_M1'
    sleep = Sleep_MSP(dataPath=dataPath)
    sleep.save_processed_data(update_path=path, subjects=[0])
    data = sleep.get_processed_data(update_path=path, subjects=[0])
    labels, read_datas = data[0], data[1]
    print(read_datas)
