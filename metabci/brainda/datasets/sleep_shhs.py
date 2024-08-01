# -*- coding: utf-8 -*-
# Organization: Hangzhou MindMatrixes Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/4
# License: MIT License

import ntpath
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union, List, Dict, Optional
from metabci.brainda.datasets.base import BaseDataset
import glob
from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF
import xml.etree.ElementTree as ET

"""
This dataset is a non-public dataset.
if you need this dataset, please contact this mailbox:
xingjian.zhang@mindmatrixes.com
or download it here:
https://pan.baidu.com/s/1_3umu61B8wln_MylCjoDbw?pwd=mdsk
"""


class Sleep_SHHS(BaseDataset):
    """
        This is the class for the SHHS dataset and contains functions for getting and reading raw data and
        label, data processing and saving, reading processed data.
        Methods:
        save_processed_data(subjects,select_ch,update_path):
            For the original dataset that has been stored, the original dataset is processed and saved as npz file
            in the specified path.
        get_processed_data(subjects,select_ch,update_path):
            Read the processed data file,return [labels, datas]

        Dataset from:
        S. Quan et al., “The Sleep Heart Health Study: design, rationale, and methods,” Sleep,
        vol. 20, no. 12, pp. 1077–1085, Dec. 1997, doi: 10.1093/sleep/20.12.1077.

        The Sleep Heart Health Study (SHHS) is a multi-center cohort study implemented by the National Heart Lung &
        Blood Institute to determine the cardiovascular and other consequences of sleep-disordered breathing. It tests
        whether sleep-related breathing is associated with an increased risk of coronary heart disease, stroke, all
        cause mortality, and hypertension.  In all, 6,441 men and women aged 40 years and older were enrolled between
        November 1, 1995 and January 31, 1998 to take part in SHHS Visit 1. During exam cycle 3 ( 2001- 2003), a second
        polysomnogram (SHHS Visit 2) was obtained in 3,295 of the participants. CVD Outcomes data were monitored and
        adjudicated by parent cohorts between baseline and 2011.
    """

    _EVENTS = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "R": 4
    }

    _CHANNELS = [
        "EEG",
        "EEG(sec)",
        "EOG(L)",
        "EOG(R)",
        "ECG",
        "EMG"
    ]

    def __init__(self, dataPath: str = None):
        self.dataPath = dataPath
        super().__init__(
            dataset_code="shhs",
            subjects=list(range(49)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm='shhs'
        )

    def data_path(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        """If you already have a local file and know the path to it
        Parameters
        ----------

        Returns
        -------
        str
            local path of the target file
        """

        #  input your path of sleep-telemetry folder
        dataPath = self.dataPath
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")
        psg_fnames = glob.glob(os.path.join(dataPath, "*.edf"))  # eeg date
        subject_id = int(subject)
        path = psg_fnames[subject_id]

        return [[path]]

    def label_path(
            self, subject: Union[str, int]) -> List[List[Union[str, Path]]]:
        dataPath = self.dataPath
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        ann_fnames = glob.glob(os.path.join(dataPath, "*nsrr.xml"))  # label
        subject_id = int(subject)
        path = ann_fnames[subject_id]

        return [[path]]

    def _get_single_subject_label(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Get label data for a single subject.

        Parameters
        ----------
        subject : Union[str, int]
            Subject ID as a string or integer.
        verbose : Optional[Union[bool, str, int]]
            Verbosity level.

        Returns
        -------
        Dict[str, Dict[str, str]]
            A dictionary where each key is a session and each value is a dictionary of labels.
        """

        dests = self.label_path(subject)
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                if not run_file.endswith('.edf'):
                    raw, raw_bool = self.readAnnotFiles(run_file)
                    runs["run_{:d}".format(irun)] = raw, raw_bool
                sess["session_{:d}".format(isess)] = runs
            sess[f"session_{isess}"] = runs
        print(f'load subject label :{subject}')
        return sess

    def _get_single_subject_data(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> dict[str, dict[str, RawEDF]]:

        dests = self.data_path(subject)
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                run_file = run_file
                raw = read_raw_edf(run_file, preload=True, verbose=False, stim_channel=None)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        print(f'load subject data :{subject}')
        return sess

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
            select_ch = ["EEG"]
        self.update_path = update_path
        if subjects is None:
            subjects = self.subjects
        EPOCH_SEC_SIZE = 30
        raws = super().get_data(subjects)
        anns = self._get_label(subjects)
        sampling_rate = 100
        savePath_ch = update_path
        for idx, subject in enumerate(subjects):
            rawdata = raws[subject]['session_0']['run_0']
            if rawdata.info['sfreq'] != sampling_rate:
                rawdata = rawdata.resample(sfreq=sampling_rate)
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

            channel_data = rawdata.to_data_frame()[select_ch]
            channel_data.set_index(np.arange(len(channel_data)))
            if data_idx[-1] > len(channel_data) - 1:
                deleteIndx = data_idx[-1] - (len(channel_data) - 1)
                deleteIndxEpoch = int(deleteIndx // (EPOCH_SEC_SIZE * sampling_rate))
                deleteIndxEpoch_remain = int(deleteIndx % (EPOCH_SEC_SIZE * sampling_rate))

                if deleteIndxEpoch_remain == 0:
                    labels = labels[:-deleteIndxEpoch]
                    data_idx = data_idx[:-deleteIndx]
                else:
                    deleteIndxEpoch = deleteIndxEpoch + 1
                    labels = labels[:-deleteIndxEpoch]
                    deleteIndxRaw = deleteIndx + int(EPOCH_SEC_SIZE * sampling_rate - deleteIndxEpoch_remain)
                    data_idx = data_idx[:-deleteIndxRaw]
                print("EDF数据比标签数据短, 删除最后{}个epoch".format(deleteIndxEpoch))

            channel_data = channel_data.values[data_idx]
            if len(channel_data) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("原始数据不能被30S整除，有问题")
            n_epochs = int(len(channel_data) / (EPOCH_SEC_SIZE * sampling_rate))
            data_X = np.asarray(np.split(channel_data, n_epochs)).astype(np.float32)
            lable_y = labels.astype(np.int32)
            assert len(data_X) == len(lable_y)
            del channel_data
            filename = ntpath.basename(self.data_path(subject)[0][0]).split("/")[-1].replace(".edf", ".npz")
            save_dict = {
                "x": data_X,
                "y": lable_y,
                "fs": sampling_rate,
            }
            np.savez(os.path.join(savePath_ch, filename), **save_dict)
            del data_X
            del save_dict

    def get_processed_data(self,
                           subjects: List[Union[int, str]] = None,
                           force_update: bool = False,
                           update_path: Optional[Union[str, Path]] = None,
                           proxies: Optional[Dict[str, str]] = None,
                           verbose: Optional[Union[bool, str, int]] = None,
                           num_classes: Optional[int] = 5) \
            -> list:
        """
            Read data from .npz files saved in the specified path for specified subjects.

            Parameters
            ----------
            subjects : list, optional
                List of subject indices to read data from. If None, reads data for all subjects.
            update_path : str, optional
                Path to the directory containing .npz files. You can leave this value out
                if you're already using the function

            Returns
            -------
            [labels,read_datas]
            labels : ndarray
                Concatenated labels array from all specified subjects.
            read_datas : ndarray
                Concatenated data array from all specified subjects, each subject's data as one segment.

            Notes
            -----
            This function loads data from .npz files located in `update_path` for the specified `subjects`.
            It concatenates the 'y' (labels) and 'x' (data) arrays from each .npz file into `labels` and `read_datas`
            respectively.
            """
        if update_path is None:
            update_path = self.update_path
        if subjects is None:
            subjects = self.subjects
        res_fnames = glob.glob(os.path.join(update_path, "*.npz"))
        res_fnames = np.asarray(res_fnames)
        read_datas = None
        labels = None
        for id in subjects:

            data = np.load(res_fnames[id])
            label = data['y']
            read_data = data['x']
            if read_datas is None:
                read_datas = read_data
                labels = label
            else:
                read_datas = np.concatenate((read_datas, read_data), axis=0)
                labels = np.concatenate((labels, label), axis=0)

        if num_classes == 2:
            labels = [0 if label == 0 else 1 for label in labels]

        if num_classes == 3:
            labels = [0 if label == 0 else 2 if label == 4 else 1 for label in labels]

        if num_classes == 4:
            labels = [0 if label == 0 else 1 if label in [1, 2] else 2 if label == 3 else 3 for label in labels]
        read_datas = read_datas.transpose(0, 2, 1)
        labels = np.array(labels)
        return [labels, read_datas]

    def _get_label(
            self,
            subjects: List[Union[int, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> Dict[Union[int, str], Dict[str, Dict[str, str]]]:
        """
        Get label data.

        Parameters
        ----------
        subjects : List[Union[int, str]]
            Subjects whose label data should be returned.

        Returns
        -------
        Dict[Union[int, str], Dict[str, Dict[str, str]]]
            Returned label data, structured as
            {
                subject_id: {'session_id': {'run_id': label_data}}
            }

        Raises
        ------
        ValueError
            Raised if a subject is not valid.
        """
        # Use default subjects if not provided
        if subjects is None:
            subjects = self.subjects

        labels = dict()
        for subject in subjects:
            if subject not in self.subjects:
                raise ValueError(f"Invalid subject {subject} given")
            labels[subject] = self._get_single_subject_label(subject, verbose)

        return labels

    @staticmethod
    def readAnnotFiles(path_label):
        """读取数据标签，同时如果有额外标签，返回的标志位置true"""
        flag_del = False
        tree = ET.parse(path_label)
        root = tree.getroot()
        # 定义表格的列
        columns = ["EventConcept", "Start", "Duration"]
        # 创建一个空的列表
        data = []
        # 遍历XML中的ScoredEvents
        for scored_event in root.findall(".//ScoredEvent"):
            event_data = {}
            # 遍历ScoredEvent中的子元素并添加到字典
            for element in scored_event:
                event_data[element.tag] = element.text
            # 只保留 EventType 为 Stages|Stages 的行
            if event_data.get("EventType") == "Stages|Stages":
                # stage的名称替换
                event_concept = event_data.get("EventConcept", "")
                if event_concept == "Wake|0":
                    event_data["EventConcept"] = "W"
                elif event_concept == "Stage 1 sleep|1":
                    event_data["EventConcept"] = "N1"
                elif event_concept == "Stage 2 sleep|2":
                    event_data["EventConcept"] = "N2"
                elif event_concept == "Stage 3 sleep|3":
                    event_data["EventConcept"] = "N3"
                elif event_concept == "Stage 4 sleep|4":
                    event_data["EventConcept"] = "N3"
                elif event_concept == "REM sleep|5":
                    event_data["EventConcept"] = "R"
                else:
                    print("存在名称问题:{},该文件跳过处理".format(event_data["EventConcept"]))
                    flag_del = True
                data.append(event_data)
        # 使用列表创建DataFrame
        df = pd.DataFrame(data, columns=columns)

        return df, flag_del


if __name__ == "__main__":
    path = r'D:\sleep-data\SHHS\raw'             # 原始数据raw_data存储地址
    dataPath = r'D:\sleep-data\SHHS\npz'         # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)

    subjects = [0, 1, 2]                         # None则代表处理所有被试
    select_ch = ["EEG", "EOG(L)"]                # None则代表使用单通道"EEG"
    num_classes = 2                              # 睡眠分期的分类任务，支持2-5类

    sleep = Sleep_SHHS(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete.")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print("labels.size: " + str(labels.size))
    print("datas.shape: " + str(read_datas.shape))
