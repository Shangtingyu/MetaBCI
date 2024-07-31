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
from metabci.brainda.datasets.sleep_shhs import Sleep_SHHS
import glob

"""
This dataset is a non-public dataset.
if you need this dataset, please contact this mailbox:
xingjian.zhang@mindmatrixes.com
or download it here:
https://pan.baidu.com/s/1Ery-1gP4PHIX1mRANRasCA?pwd=mdsk 
"""


class Sleep_Apples(Sleep_SHHS):
    """
    Dataset from:
    Zhang GQ, Cui L, Mueller R, Tao S, Kim M, Rueschman M, Mariani S, Mobley D, Redline S.
    The National Sleep Research Resource: towards a sleep data commons. J Am Med Inform Assoc.
    2018 Oct 1;25(10):1351-1358. doi: 10.1093/jamia/ocy064. PMID: 29860441; PMCID: PMC6188513.
    https://sleepdata.org/datasets/nfs
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
        self.dataset_code = "shhs"
        self.events = self._EVENTS
        self.channels = self._CHANNELS
        super().__init__(dataPath=dataPath)

    @staticmethod
    def readAnnotFiles(path):
        dataTable = pd.read_table(path, header=None)
        dataTable = dataTable.drop([1, 2, 5], axis=1)  # 删除多余的列（已确认是无用信息行）
        keepClassNames = ['W', 'R', 'N1', 'N2', 'N3']
        condition = dataTable[0].isin(keepClassNames)
        filteredDataTable = dataTable[condition]  # 只保留符合条件的行

        return filteredDataTable

    @staticmethod
    def checkTime(time_edf, time_ann):
        """判断标签和数据的时间是否一致，不一致调整数据的开始时间"""
        onsetSec = 0
        flag_del = False
        # 该日期部分可以是任意日期，因为我们不关心日期，只是为了获取时间差
        fixed_date = "2023-01-01"
        # 将时间字符串解析为 datetime 对象，合并到固定日期
        time_edf = datetime.strptime(fixed_date + " " + time_edf, "%Y-%m-%d %H:%M:%S")
        time_ann = datetime.strptime(fixed_date + " " + time_ann, "%Y-%m-%d %H:%M:%S")

        # 创建12点的 datetime 对象
        thresholdTime = datetime(time_edf.year, time_edf.month, time_edf.day, 12, 0, 0)

        # 可能会从12点之后才开始计算时间, 判断是否过夜了
        if time_edf < thresholdTime:
            time_edf += timedelta(days=1)
        if time_ann < thresholdTime:
            time_ann += timedelta(days=1)

        if time_edf == time_ann:
            pass
        elif time_edf < time_ann:
            timeDifference = time_ann - time_edf
            differentSecond = timeDifference.total_seconds()
            onsetSec = differentSecond  # 使用EDF的开始时间
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
        # 可能存在结尾EDF数据比标签数据短的情况（数据损坏导致的？）
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

        raw_ch = raw_ch_df.values[data_idx]  # 从原始数据中选择保留的indx对应的数值

        # 再次验证数据能被30-s整除 epochs
        if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
            raise Exception("原始数据不能被30S整除，有问题")

        n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))
        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        # 确保数据和标签是对应的
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
        if select_ch is None:
            select_ch = ["C3_M2"]
        annotFiles = []
        list_200HZ = []
        raws = self.get_data(subjects)
        for i in subjects:
            annotFiles.append(self.label_path(i))
        for idx, subject in enumerate(subjects):
            rawdata = raws[subject]['session_0']['run_0']
            annotdata = self.readAnnotFiles(annotFiles[idx][0][0])  # 类型是frame
            sampling_rate = int(rawdata.info['sfreq'])
            # 跳过采样率为200Hz
            if sampling_rate != 100:
                list_200HZ.append(subject)
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
            savePath_ch = update_path + '/'
            for idx, ch in enumerate(select_ch):
                if idx == 0:
                    savePath_ch = savePath_ch + ch  # 第一次循环，不加'-'
                else:
                    savePath_ch = savePath_ch + '-' + ch  # 后续循环，加'-'
            print(f'save path: {savePath_ch}')
            if not os.path.exists(savePath_ch):
                os.makedirs(savePath_ch)
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
    dataPath = r'D:\sleep-data\Apples\raw'
    path = r'D:\sleep-data\Apples\raw\C3_M2-ROC-LOC'
    sleep = Sleep_Apples(dataPath=dataPath)
    sleep.save_processed_data(update_path=dataPath, select_ch=["C3_M2", "ROC", "LOC"])
    data = sleep.get_processed_data(update_path=path, subjects=[0])
    labels, read_datas = data[0], data[1]
    print(read_datas)
