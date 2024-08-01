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
from metabci.brainda.datasets.sleep_apples import Sleep_Apples
from typing import Union, List, Dict, Optional


class Sleep_MSP(Sleep_Apples):
    """
        This is the class for the MSP dataset and contains functions for getting and reading raw data and
        label, data processing and saving, reading processed data.
        Methods:
        save_processed_data(subjects,select_ch,update_path):
            For the original dataset that has been stored, the original dataset is processed and saved as npz file
            in the specified path.
        get_processed_data(subjects,select_ch,update_path):
            Read the processed data file,return [labels, datas]

        Dataset from:
        DiPietro JA, Raghunathan RS, Wu HT, Bai J, Watson H, Sgambati FP, Henderson JL, Pien GW. Fetal heart rate
        during maternal sleep. Dev Psychobiol. 2021 Jul;63(5):945-959. doi: 10.1002/dev.22118. Epub 2021 Mar 25.
        PMID: 33764539.

        The Maternal Sleep in Pregnancy and the Fetus (MSP) data were generated by standard overnight laboratory-based
        polysomnography (PSG) that included an additional monitor for detecting and quantifying the fetal
        electrocardiogram (ECG), conducted on 106 women during the 36th week of pregnancy. The primary aims of this
        study were to: 1. evaluate whether episodes of maternal sleep disordered breathing contemporaneously exert
        deleterious influences on the fetus, as indicated by fetal heart rate and variability, the primary metrics for
        evaluating antepartum well-being; and 2. describe fetal heart rate and variability in relation to maternal sleep
        stages. A secondary aim was to evaluate whether fetuses of women with higher levels of sleep disordered
        breathing exhibited indicators of reduced neuromaturation.
    """

    _EVENTS = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "R": 4
    }

    _CHANNELS = [
        "C3_M2",
        "C4_M1",
        "F3_M2",
        "F4_M1",
        "O1_M2",
        "O2_M1",
        "ROC",
        "LOC",
        "ROC",
        "ECG",
        "EMG1",
        "EMG2"
    ]

    def __init__(self, dataPath: str = None):
        """
            Args:
                dataPath (str): Target storage address for raw data edf
        """
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
        sampling_rate = 100
        annotFiles = []
        for i in subjects:
            annotFiles.append(self.label_path(i))
            print(f'load subject label :{i}')
        for idx, subject in enumerate(subjects):
            rawdata = raws[subject]['session_0']['run_0']
            if rawdata.info['sfreq'] != sampling_rate:
                rawdata = rawdata.resample(sfreq=sampling_rate)
            annotdata = self.readAnnotFiles(annotFiles[idx][0][0])

            raw_startTime = str(rawdata.info['meas_date']).split(" ")[1]
            raw_startTime = raw_startTime.split("+")[0]
            ann_startTime = annotdata.iloc[0, 1]

            onsetSec, flag_del = self.checkTime(raw_startTime, ann_startTime)
            if flag_del:
                continue

            durationSecond = len(annotdata) * 30
            labels = annotdata.iloc[:, 0].to_numpy()
            mapping_function = np.frompyfunc(self._EVENTS.get, 1, 1)
            labels = mapping_function(labels)
            data_idx = int(onsetSec * sampling_rate) + np.arange(durationSecond * sampling_rate, dtype=int)
            data_X, lable_y = self.cleanData(rawdata, labels, data_idx, select_ch, sampling_rate)
            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            save_dict = {
                "x": data_X,
                "y": lable_y,
                "fs": sampling_rate,
                "ch_label": select_ch,
            }
            np.savez(os.path.join(savePath_ch, filename), **save_dict)


if __name__ == "__main__":
    path = r'D:\sleep-data\msp\edfs'           # 原始数据raw_data存储地址
    dataPath = r'D:\sleep-data\msp\F4_M1'       # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)

    subjects = [0, 1, 2]                      # None则代表处理所有被试
    select_ch = ["C3_M2", "F4_M1"]            # None则代表使用单通道"F4_M1"
    num_classes = 4                           # 睡眠分期的分类任务，支持2-5类

    sleep = Sleep_MSP(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete.")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print("labels.size: " + str(labels.size))
    print("datas.shape: " + str(read_datas.shape))
