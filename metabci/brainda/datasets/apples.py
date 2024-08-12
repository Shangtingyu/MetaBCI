# -*- coding: utf-8 -*-
# Organization: Hangzhou MindMatrixes Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/14
# License: MIT License

import ntpath
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union, List, Dict, Optional
from metabci.brainda.datasets.shhs import SHHS
import glob

"""
This dataset is a non-public dataset.
if you need this dataset, please contact this mailbox:
xingjian.zhang@mindmatrixes.com
or download it here:
https://pan.baidu.com/s/12OU1h5rJR0yK8jkk8B7FvQ?pwd=mdsk 
"""


class Apples(SHHS):
    """
        This is the class for the Apples dataset and contains functions for getting and reading raw data and
        label, data processing and saving, reading processed data.
        Methods:
        save_processed_data(subjects,select_ch,update_path):
            For the original dataset that has been stored, the original dataset is processed and saved as npz file
            in the specified path.
        get_processed_data(subjects,select_ch,update_path):
            Read the processed data file,return [labels, datas]

        Dataset from:
        S. F. Quan et al., “The Association between Obstructive Sleep Apnea and Neurocognitive Performance—The Apnea
        Positive Pressure Long-term Efficacy Study (APPLES),” Sleep, vol. 34, no. 3, pp. 303–314, Mar. 2011,
        doi: 10.1093/sleep/34.3.303.

        The Apnea Positive Pressure Long-term Efficacy Study (APPLES) was a NHLBI-sponsored 6-month, randomized,
        double-blind, 2-arm, sham-controlled, multicenter trial conducted at 5 U.S. university hospitals, or private
        practices. 1,516 participants were enrolled since November 2003 and studied for up to 6 months over 11 visits,
        of which 1,105 were randomized to active vs. sham CPAP (REMstar Pro, Philips Respironics, Inc.) devices; the
        sham CPAP device closely simulates the airflow through the exhalation port and the operating noise of the active
        CPAP device. 1,098 participants were diagnosed with OSA contributed to the analysis of the primary outcome
        measures. The study was completed in August 2008.
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
        "O1_M2",
        "O2_M1",
        "ROC",
        "LOC",
        "ROC",
        "ECG",
        "EMG"
    ]

    def __init__(self, dataPath: str = None, subjects=None):
        """
            Args:
                dataPath (str): Target storage address for raw data edf
                subjects (list): List of subject numbers,defaults to all subjects
        """
        super().__init__(dataPath=dataPath, subjects=subjects)
        if subjects is None:
            subjects = list(range(50))
        self.dataPath = dataPath
        self.dataset_code = "apples"
        self.events = self._EVENTS
        self.channels = self._CHANNELS
        if subjects is None:
            subjects = list(range(50))
        self.subjects = subjects
        self.strate = 100
        self.paradigm = "sleep stage"


    @staticmethod
    def readAnnotFiles(path):
        dataTable = pd.read_table(path, header=None)
        dataTable = dataTable.drop([1, 2, 5], axis=1)
        keepClassNames = ['W', 'R', 'N1', 'N2', 'N3']
        condition = dataTable[0].isin(keepClassNames)
        filteredDataTable = dataTable[condition]

        return filteredDataTable

    @staticmethod
    def checkTime(time_edf, time_ann):

        onsetSec = 0
        flag_del = False
        # This date part can be any date, because we don't care about the date, just to get the time difference
        fixed_date = "2023-01-01"
        time_edf = datetime.strptime(fixed_date + " " + time_edf, "%Y-%m-%d %H:%M:%S")
        time_ann = datetime.strptime(fixed_date + " " + time_ann, "%Y-%m-%d %H:%M:%S")
        thresholdTime = datetime(time_edf.year, time_edf.month, time_edf.day, 12, 0, 0)

        if time_edf < thresholdTime:
            time_edf += timedelta(days=1)
        if time_ann < thresholdTime:
            time_ann += timedelta(days=1)

        if time_edf == time_ann:
            pass
        elif time_edf < time_ann:
            timeDifference = time_ann - time_edf
            differentSecond = timeDifference.total_seconds()
            onsetSec = differentSecond
            print("EDF比标签开始的时间早")
            print(time_edf)
            print(time_ann)
        elif time_edf > time_ann:
            flag_del = True
            print("EDF比标签开始的时间晚，这种情况很不寻常，估计数据有问题")
            print(time_edf)
            print(time_ann)

        return onsetSec, flag_del

    @staticmethod
    def cleanData(rawdata, labels, data_idx, select_ch, sampling_rate):
        raw_ch_df = rawdata.to_data_frame()[select_ch]
        EPOCH_SEC_SIZE = 30
        if data_idx[-1] > len(raw_ch_df) - 1:
            deleteIndx = data_idx[-1] - (len(raw_ch_df) - 1)
            deleteIndxEpoch = int(deleteIndx // (EPOCH_SEC_SIZE * sampling_rate))  # 取整
            deleteIndxEpoch_remain = int(deleteIndx % (EPOCH_SEC_SIZE * sampling_rate))  # 取余

            if deleteIndxEpoch_remain == 0:
                labels = labels[:-deleteIndxEpoch]
                data_idx = data_idx[:-deleteIndx]
            else:
                deleteIndxEpoch = deleteIndxEpoch + 1
                labels = labels[:-deleteIndxEpoch]
                deleteIndxRaw = deleteIndx + int(EPOCH_SEC_SIZE * sampling_rate - deleteIndxEpoch_remain)
                data_idx = data_idx[:-deleteIndxRaw]
            print("EDF数据比标签数据短, 删除最后{}个epoch".format(deleteIndxEpoch))

        raw_ch = raw_ch_df.values[data_idx]

        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("原始数据不能被30S整除，有问题")

        n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        return x, y

    def label_path(self, subject: Union[str, int]) -> List[List[Union[str, Path]]]:
        dataPath = self.dataPath
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        ann_fnames = glob.glob(os.path.join(dataPath, "*.annot"))  # label
        subject_id = int(subject)
        path = ann_fnames[subject_id]

        return [[path]]

    def save_processed_data(self,
                            subjects: List[Union[int, str]] = None,
                            path: Optional[Union[str, Path]] = None,
                            force_update: bool = False,
                            update_path: Optional[Union[str, Path]] = None,
                            proxies: Optional[Dict[str, str]] = None,
                            verbose: Optional[Union[bool, str, int]] = None,
                            select_ch: List[str] = None):
        """
        Save processed EEG data and corresponding labels as .npz files.

        Parameters:
        -----------
        subjects : list or None, optional
            List of subject indices to process. If None, processes all subjects.
        select_ch : list[str], optional
            Channel name(EEG, EOG(L) or EOG(R)) to select from the raw EEG data. If None, select_ch is ["EEG Fpz-Cz"]
        update_path : str or
            Path to save the processed data files. If None, defaults to a specified path.

        Notes:
        ------
        This function processes raw EEG data and its corresponding annotations (labels), applies
        filtering, and saves the segmented data into .npz files. It handles synchronization between
        raw data and annotations, filters out unwanted epochs, and ensures data integrity before saving.

        """
        if select_ch is None:
            select_ch = ["C3_M2"]
        if subjects is None:
            subjects = self.subjects
        sampling_rate = self.strate
        annotFiles = []

        for i in subjects:
            annotFiles.append(self.label_path(i))
            print(f'load subject label :{i}')
        for idx, subject in enumerate(subjects):
            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            file_path = os.path.join(update_path, filename)
            if os.path.exists(file_path):
                print(f"processed file subject{subject} already exists,pass")
                continue
            rawdata = self._get_single_subject_data(subject=subject)['session_0']['run_0']
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
            savePath_ch = update_path
            data_X, lable_y = self.cleanData(rawdata, labels, data_idx, select_ch, sampling_rate)
            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            save_dict = {
                "x": data_X,
                "y": lable_y,
                "fs": sampling_rate,
                "ch_label": select_ch,
            }
            np.savez(os.path.join(savePath_ch, filename), **save_dict)
            print(f'successfully process subject:{subject}')


if __name__ == "__main__":
    path = r'D:\sleep-data\Apples\raw'           # 原始数据raw_data存储地址
    dataPath = r'D:\sleep-data\Apples\raw\C3_M2-ROC-LOC'       # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)

    subjects = None                        # None则代表处理所有被试
    select_ch = ["C3_M2", "ROC", "LOC"]          # None则代表使用单通道"C3_M2"
    num_classes = 2                              # 睡眠分期的分类任务，支持2-5类

    sleep = Apples(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete.")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print("labels.size: " + str(labels.size))
    print("datas.shape: " + str(read_datas.shape))
