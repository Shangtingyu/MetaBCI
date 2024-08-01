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
https://pan.baidu.com/s/1WBZIOO4-D95SK7KJw_6YCA?pwd=mdsk
"""


class Sleep_mros(Sleep_SHHS):
    """
    Dataset from:
    Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S.
    The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc.
    2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064. PMID: 29860441; PMCID: PMC6188513.
    https://sleepdata.org/datasets/nfs

    """
    EPOCH_SEC_SIZE = 30
    _EVENTS = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "R": 4
    }
    _CHANNELS = ["EEG", "EOG(L)", "EOG(R)"]

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
            select_ch = ["C3-A2", "C3-A1"]
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
        print(result)
        for subject in subjects:
            # raw_info = read_raw_edf(psgFiles[i], preload=False, verbose=False).info
            # sampling_rate = raw_info['sfreq']
            #
            # list_otherHZ = []
            # if sampling_rate != 1024:
            #     list_otherHZ.append(i)
            #     print("这个文件采样率为{}".format(sampling_rate))
            #     continue
            rawdata = raws[subject]['session_0']['run_0']
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
            print(diff)
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

            # 可能存在结尾EDF数据比标签数据短的情况（数据损坏导致的？）
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

            raw_ch2_df = raw_ch2_df.values[data_idx]  # 从原始数据中选择保留的indx对应的数值
            del data_idx
            if len(raw_ch2_df) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("原始数据不能被30S整除，有问题")

            n_epochs = int(len(raw_ch2_df) / (EPOCH_SEC_SIZE * sampling_rate))
            x_ch2 = np.asarray(np.split(raw_ch2_df, n_epochs)).astype(np.float32)
            y = labels.astype(np.int32)

            # 确保数据和标签是对应的
            assert len(x_ch2) == len(y)

            del annotFrame
            del resultFrame
            del labels

            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            save_dict_2CH = {
                "x": x_ch2,
                "y": y,
            }
            np.savez(os.path.join(update_path, filename), **save_dict_2CH)


if __name__ == "__main__":
    dataPath = r'D:\sleep-data\mros\edfs'
    path = r'D:\sleep-data\mros\1'
    sleep = Sleep_mros(dataPath=dataPath)
    sleep.save_processed_data(update_path=path, subjects=[0])
    data = sleep.get_processed_data(update_path=path, subjects=[0])
    labels, read_datas = data[0], data[1]
    print(read_datas)
