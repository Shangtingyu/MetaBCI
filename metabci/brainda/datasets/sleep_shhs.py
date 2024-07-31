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
https://pan.baidu.com/s/1WBZIOO4-D95SK7KJw_6YCA?pwd=mdsk
"""


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
                print("存在名称问题:{},该文件跳过处理".format(event_data["EventConcept"]))  # 例如Unscored
                flag_del = True
            data.append(event_data)
    # 使用列表创建DataFrame
    df = pd.DataFrame(data, columns=columns)

    return df, flag_del


class Sleep_SHHS(BaseDataset):
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
        print(f'load subject :{subject_id}')
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
                # Ensure .edf suffix is present
                if not run_file.endswith('.edf'):
                    raw, _ = readAnnotFiles(run_file)
                    runs["run_{:d}".format(irun)] = raw
                sess["session_{:d}".format(isess)] = runs
            sess[f"session_{isess}"] = runs

        return sess

    def _get_single_subject_data(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> dict[str, dict[str, RawEDF]]:

        dests = self.data_path(subject)
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                # run_file = run_file.with_suffix('.edf')
                run_file = run_file
                raw = read_raw_edf(run_file, preload=True, verbose=False, stim_channel=None)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
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
        sampling_rate = 100
        annotFiles = []
        for i in subjects:
            annotFiles.append(self.label_path(i))
        for idx, subject in enumerate(subjects):
            rawdata = raws[subject]['session_0']['run_0']
            rawdata = rawdata.resample(sfreq=sampling_rate)
            annotFrame, flag_del = readAnnotFiles(annotFiles[idx][0][0])
            if flag_del:
                # 有未知标签跳过
                continue
            result_df = annotFrame[['EventConcept']].copy()
            annotFrame['Duration'] = annotFrame['Duration'].astype(float)
            annotFrame['Chunks'] = (annotFrame['Duration'] / 30).astype(int)
            result_df = result_df.loc[result_df.index.repeat(annotFrame['Chunks'])]  # 作用是根据每行的 'Chunks' 列的值，重复对应行的数据
            result_df.reset_index(drop=True, inplace=True)  # 重新设置索引
            resultFrame = result_df
            labels = resultFrame.iloc[:, 0].to_numpy()
            mapping_function = np.frompyfunc(self._EVENTS.get, 1, 1)  # 将数组元素映射为对应的值
            labels = mapping_function(labels)
            durationSecond = len(labels) * 30
            data_idx = np.arange(durationSecond * sampling_rate, dtype=int)
            print("开始处理通道：{}".format(select_ch))
            savePath_ch = update_path + '/'
            for id, ch in enumerate(select_ch):
                if id == 0:
                    savePath_ch = savePath_ch + ch  # 第一次循环，不加'-'
                else:
                    savePath_ch = savePath_ch + '-' + ch  # 后续循环，加'-'
            if not os.path.exists(savePath_ch):
                os.makedirs(savePath_ch)

            channel_data = rawdata.to_data_frame()[select_ch]
            channel_data.set_index(np.arange(len(channel_data)))  # 设置为一个新的整数索引.

            # 可能存在结尾EDF数据比标签数据短的情况（数据损坏导致的？）
            if data_idx[-1] > len(channel_data) - 1:
                deleteIndx = data_idx[-1] - (len(channel_data) - 1)
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

            channel_data = channel_data.values[data_idx]  # 从原始数据中选择保留的indx对应的数值

            # 再次验证数据能被30-s整除 epochs
            if len(channel_data) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("原始数据不能被30S整除，有问题")

            n_epochs = int(len(channel_data) / (EPOCH_SEC_SIZE * sampling_rate))
            data_X = np.asarray(np.split(channel_data, n_epochs)).astype(np.float32)
            lable_y = labels.astype(np.int32)
            # 确保数据和标签是对应的
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
        print(f'load file: {res_fnames}')
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
            labels = np.array(labels)
        if num_classes ==2:
            # 将1,2,3,4视为一类，区分0，变成二分类任务
            labels = [0 if label == 0 else 1 for label in labels]

        if num_classes ==3:
            # 将1,2,3视为一类，区分0和4，变成三分类任务
            labels = [0 if label == 0 else 2 if label == 4 else 1 for label in labels]

        if num_classes ==4:
            # 将1,2视为一类，区分0和3和4，变成四分类任务
            labels = [0 if label == 0 else 1 if label in [1, 2] else 2 if label == 3 else 3 for label in labels]
        read_datas = read_datas.transpose(0, 2, 1)
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


if __name__ == "__main__":
    path = r'D:\sleep-data\shhs\edfs'
    dataPath = r'D:\sleep-data\shhs'
    sleep = Sleep_SHHS(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, select_ch=["EEG", "EOG(L)"],subjects=[0,1,2,3])
    savepath = r'D:\sleep-data\shhs\EEG-EOG(L)'
    data = sleep.get_processed_data(subjects=[0], update_path=savepath)
    labels, read_datas = data[0], data[1]
    print(read_datas)
