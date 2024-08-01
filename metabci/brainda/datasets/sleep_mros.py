# -*- coding: utf-8 -*-
# Organization: Hangzhou MindMatrixes Technology Co.
# Authors: Shangtingyu
# Date: 2024/8/1
# License: MIT License

import ntpath
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union, List, Dict, Optional
from metabci.brainda.datasets.sleep_shhs import Sleep_SHHS



"""
This dataset is a non-public dataset.
if you need this dataset, please contact this mailbox:
xingjian.zhang@mindmatrixes.com
or download it here:
https://pan.baidu.com/s/1Tr9dj1jNJfcfNPv9H74TLA?pwd=mdsk 
"""


class Sleep_mros(Sleep_SHHS):
    """
        This is the class for the Mros dataset and contains functions for getting and reading raw data and
        label, data processing and saving, reading processed data.
        Methods:
        save_processed_data(subjects,select_ch,update_path):
            For the original dataset that has been stored, the original dataset is processed and saved as npz file
            in the specified path.
        get_processed_data(subjects,select_ch,update_path):
            Read the processed data file,return [labels, datas]

        Dataset from:
        T. Blackwell et al., “Associations Between Sleep Architecture and Sleep‐Disordered Breathing and Cognition in
        Older Community‐Dwelling Men: The Osteoporotic Fractures in Men Sleep Study,” J American Geriatrics Society,
        vol. 59, no. 12, pp. 2217–2225, Dec. 2011, doi: 10.1111/j.1532-5415.2011.03731.x.

        MrOS is an ancillary study of the parent Osteoporotic Fractures in Men Study. Between 2000 and 2002, 5,994
        community-dwelling men 65 years or older were enrolled at 6 clinical centers in a baseline examination.
        Between December 2003 and March 2005, 3,135 of these participants were recruited to the Sleep Study when they
        underwent full unattended polysomnography and 3 to 5-day actigraphy studies. The objectives of the Sleep Study
        are to understand the relationship between sleep disorders and falls, fractures, mortality, and vascular disease.
    """

    _EVENTS = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "R": 4
    }

    _CHANNELS = [
        "C3",
        "C4",
        "A1",
        "A2",
        "ROC",
        "LOC",
        "ECG L",
        "ECG R",
        "L Chin",
        "R Chin"
    ]

    def __init__(self, dataPath: str = None):

        self.dataset_code = "mros"
        self.channels = self._CHANNELS,
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
            select_ch = ["C3"]
        self.update_path = update_path
        if subjects is None:
            subjects = self.subjects
        EPOCH_SEC_SIZE = 30
        raws = super().get_data(subjects)
        anns = self._get_label(subjects)
        sampling_rate = 100
        result = []
        for string in select_ch:
            if '-' in string:
                result.append(string.split('-'))
            else:
                result.append([string])
        for subject in subjects:
            rawdata = raws[subject]['session_0']['run_0']
            if rawdata.info['sfreq'] != sampling_rate:
                rawdata = rawdata.resample(sfreq=sampling_rate)
            df = rawdata.to_data_frame()
            diff = []
            for ch in result:
                if len(ch) == 2:
                    select_ch1 = ch[0]
                    select_ch2 = ch[1]
                    diff_signal = df[select_ch1] - df[select_ch2]
                    diff_signal = diff_signal.to_frame()
                else:
                    select_ch1 = ch[0]
                    diff_signal = df[select_ch1]
                    diff_signal = diff_signal.to_frame()
                diff.append(diff_signal)
            raw_ch2_df = pd.concat(diff, axis=1)
            raw_ch2_df.set_index(np.arange(len(raw_ch2_df)))
            del diff_signal
            del df
            del rawdata
            annotFrame, flag_del = anns[subject]['session_0']['run_0']
            if flag_del:
                continue
            result_df = annotFrame[['EventConcept']].copy()
            annotFrame['Duration'] = annotFrame['Duration'].astype(float)
            annotFrame['Chunks'] = (annotFrame['Duration'] / 30).astype(int)
            result_df = result_df.loc[result_df.index.repeat(annotFrame['Chunks'])]
            result_df.reset_index(drop=True, inplace=True)
            resultFrame = result_df

            labels = resultFrame.iloc[:, 0].to_numpy()
            mapping_function = np.frompyfunc(self._EVENTS.get, 1, 1)
            labels = mapping_function(labels)

            durationSecond = len(labels) * 30
            data_idx = np.arange(durationSecond * sampling_rate, dtype=int)

            if data_idx[-1] > len(raw_ch2_df) - 1:
                deleteIndx = data_idx[-1] - (len(raw_ch2_df) - 1)
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

            raw_ch2_df = raw_ch2_df.values[data_idx]
            del data_idx
            if len(raw_ch2_df) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("原始数据不能被30S整除，有问题")

            n_epochs = int(len(raw_ch2_df) / (EPOCH_SEC_SIZE * sampling_rate))
            x_ch = np.asarray(np.split(raw_ch2_df, n_epochs)).astype(np.float32)
            y = labels.astype(np.int32)

            assert len(x_ch) == len(y)

            del annotFrame
            del resultFrame
            del labels

            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            save_dict = {
                "x": x_ch,
                "y": y,
                "fs": sampling_rate,
                "ch_label": select_ch,
            }
            np.savez(os.path.join(update_path, filename), **save_dict)


if __name__ == "__main__":
    path = r'D:\sleep-data\mros\raw'           # 原始数据raw_data存储地址
    dataPath = r'D:\sleep-data\mros\npz'       # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)

    subjects = [0, 1, 2]                      # None则代表处理所有被试
    select_ch = ["C3", "C4-A1"]               # None则代表使用单通道"C3"
    num_classes = 4                           # 睡眠分期的分类任务，支持2-5类

    sleep = Sleep_mros(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete.")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print("labels.size: " + str(labels.size))
    print("datas.shape: " + str(read_datas.shape))
