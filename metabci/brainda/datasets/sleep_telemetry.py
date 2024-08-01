# -*- coding: utf-8 -*-
# Organization: Hangzhou MindMatrixes Technology Co.
# Authors: Shangtingyu
# Date: 2024/7/4
# License: MIT License

"""
Basic Value and Label Class from Sleep-EDF Database Expanded
https://www.physionet.org/content/sleep-edfx/1.0.0/sleep-telemetry/#files-panel
"""
import ntpath
import numpy as np
import warnings
import os
from pathlib import Path
from typing import Union, List, Dict, Optional
from mne import read_annotations
from metabci.brainda.utils.download import mne_data_path
from metabci.brainda.datasets.base import BaseDataset
import glob
from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF
import math


class Sleep_telemetry(BaseDataset):
    """
    This is the class for the SleepCassette dataset and contains functions for getting and reading raw data and
    label, data processing and saving, reading processed data.
    Methods:
        save_processed_data(subjects,select_ch,update_path):
            For the original dataset that has been stored (and will be downloaded automatically if it has not
            been downloaded yet), the original dataset is processed and saved as npz file in the specified path.
        get_processed_data(subjects,select_ch,update_path):
            Read the processed data file,return [labels, datas]

    Dataset from:
    M. S. Mourtazaev, B. Kemp, A. H. Zwinderman, and H. A. C. Kamphuisen, “Age and Gender Affect Different
    Characteristics of Slow Waves in the Sleep EEG,” Sleep,
    vol. 18, no. 7, pp. 557–564, Sep. 1995, doi: 10.1093/sleep/18.7.557

    The sleep-edf database contains a lot of whole-night polygraphic sleep recordings, which include
    electroencephalography (EEG), electrooculography (EOG), electromyography (EMG) of the chin, and event markers.
    Some records also contain respiration and body temperature. The corresponding hypnograms (sleep patterns) were
    manually scored by well-trained technicians according to the Rechtschaffen and Kales manual, and are also available.

    The 44 ST* files (ST = Sleep Telemetry) were obtained in a 1994 study of temazepam effects on sleep in 22 Caucasian
    males and females without other medication. Subjects had mild difficulty falling asleep but were otherwise healthy.
    The PSGs of about 9 hours were recorded in the hospital during two nights, one of which was after temazepam intake,
    and the other of which was after placebo intake. EOG, EMG and EEG signals were sampled at 100 Hz,and the event
    marker at 1 Hz. Files are named in the form ST7ssNJ0-PSG.edf where ss is the subject number, and N is the night.
    """

    _EVENTS = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
        "Sleep stage ?": 5,
        "Movement time": 5
    }

    _CHANNELS = [
        'EEG Fpz-Cz',
        'EEG Pz-Oz',
        'EOG horizontal',
        'EMG submental'
    ]

    def __init__(self, dataPath: str = None):
        """

            Args:
                dataPath (str): Target storage address for raw data edf

        """
        self.dataPath = dataPath
        self.sleep_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/"
        super().__init__(
            dataset_code="sleep_edf",
            subjects=list(range(44)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm="sleep stage",

        )

    def data_path_url(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        """
        If you don't have this data file, you need to download it online. This process can be slow.

        Parameters
        ----------
        Returns
        -------
        str
            local path of the target file
        """
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")
        if path is None:
            path = self.dataPath
        subject_id = int(subject)
        subject_id = subject_id + 1
        psg, ann = self.read_data_name(line_number=subject_id)
        url_psg = "{:s}{:s}".format(self.sleep_URL, psg)
        file_dest = mne_data_path(
            url_psg,
            "sleep_edf",
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )

        return [[file_dest]]

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

        dataPath = self.dataPath
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")
        psg_fnames = glob.glob(os.path.join(dataPath, "*PSG.edf"))
        subject_id = int(subject)
        path = psg_fnames[subject_id]

        return [[path]]

    def _get_single_subject_data(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> dict[str, dict[str, RawEDF]]:

        try:
            dests = self.data_path(subject)
        except:
            dests = self.data_path_url(subject)
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_edf(run_file, preload=True, verbose=False, stim_channel=None)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs
        print(f'load subject data :{subject}')
        return sess

    def get_freq(self, subject: int):
        raw = self._get_single_subject_data(subject)["session_0"]["run_0"]
        sample_rating = raw.info['sfreq']
        return sample_rating

    @staticmethod
    def read_data_name(line_number):
        """
        Reads the filename from 'Hypnogram_filenames.txt' and 'psg_filenames.txt'
        at the specified line number.

        Args:
            line_number (int): The line number to read from the files.

        Returns:
            tuple: (data_list, label) where data_list is from 'psg_filenames.txt'
                   and label is from 'Hypnogram_filenames.txt'.
        """
        current_file = os.path.abspath(__file__)
        project_root = current_file

        while not os.path.isfile(os.path.join(project_root, 'setup.py')):
            parent_dir = os.path.dirname(project_root)
            if parent_dir == project_root:
                raise FileNotFoundError("setup.py not found in any parent directory")
            project_root = parent_dir

        hypnogram_file_path = os.path.join(project_root, 'docs', 'Hypnogram_filenames.txt')
        psg_file_path = os.path.join(project_root, 'docs', 'psg_filenames.txt')

        try:
            with open(hypnogram_file_path, 'r', encoding='utf-8') as file:
                for current_line_number, line in enumerate(file, start=1):
                    if current_line_number == line_number:
                        label = line.strip()
                        break
                else:
                    raise ValueError(f"Line number {line_number} not found in {hypnogram_file_path}")

            with open(psg_file_path, 'r', encoding='utf-8') as file:
                for current_line_number, line in enumerate(file, start=1):
                    if current_line_number == line_number:
                        data_list = line.strip()
                        break
                else:
                    raise ValueError(f"Line number {line_number} not found in {psg_file_path}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None, None
        except ValueError as e:
            print(f"Error: {e}")
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None, None

        return data_list, label

    def label_path_url(
            self,
            subject: Union[str, int],
            path: Optional[Union[str, Path]] = None,
            force_update: bool = False,
            update_path: Optional[bool] = None,
            proxies: Optional[Dict[str, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")

        subject_id = int(subject)
        subject_id = subject_id + 1
        psg, ann = self.read_data_name(subject_id)
        url_ann = "{:s}{:s}".format(self.sleep_URL, ann)
        file_dest = mne_data_path(
            url_ann,
            "sleep_edf",
            path=path,
            proxies=proxies,
            force_update=force_update,
            update_path=update_path,
        )
        return [[file_dest]]

    def label_path(
            self, subject: Union[str, int]) -> List[List[Union[str, Path]]]:

        dataPath = self.dataPath
        if subject not in self.subjects:
            raise ValueError("Invalid subject id")
        ann_fnames = glob.glob(os.path.join(dataPath, "*Hypnogram.edf"))
        subject_id = int(subject)
        path = ann_fnames[subject_id]
        return [[path]]

    def _get_single_subject_label(
            self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> dict[str, dict[str, RawEDF]]:
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
        try:
            dests = self.label_path(subject)
        except Exception as e:
            dests = self.label_path_url(subject)
        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_file in enumerate(run_dests):
                raw = read_raw_edf(run_file, preload=True, verbose=False, stim_channel=None)
                runs["run_{:d}".format(irun)] = raw
                sess["session_{:d}".format(isess)] = runs
            sess[f"session_{isess}"] = runs
        print(f'load subject label :{subject}')
        return sess

    def _get_label(
            self,
            subjects: List[Union[int, str]] = None,
            verbose: Optional[Union[bool, str, int]] = None,
    ) -> dict[int | str, dict[str, dict[str, RawEDF]]]:
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
            Channel name(EEG Fpz-Cz, EEG Pz-Oz or EOG horizontal) to select from the raw EEG data. If None, select_ch is ["EEG Fpz-Cz"]
        update_path : str or
            Path to save the processed data files. If None, defaults to a specified path.

        Notes:
        ------
        This function processes raw EEG data and its corresponding annotations (labels) and saves the segmented data into .npz files. It handles synchronization between
        raw data and annotations, filters out unwanted epochs, and ensures data integrity before saving.

        """

        if select_ch is None:
            select_ch = ["EEG Fpz-Cz"]
        self.update_path = update_path
        if subjects is None:
            subjects = self.subjects
        EPOCH_SEC_SIZE = 30
        raws = self.get_data(subjects)
        anns = self._get_label(subjects)

        for subject in subjects:
            raw = raws[subject]['session_0']['run_0']
            sampling_rate = raw.info['sfreq']
            raw_ch_df = raw.to_data_frame()[select_ch]
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            ann = anns[subject]['session_0']['run_0']
            warnings.resetwarnings()
            raw_start_time = raw.info['meas_date'].strftime("%Y-%m-%d %H:%M:%S UTC")
            ann_start_time = ann.info['meas_date'].strftime("%Y-%m-%d %H:%M:%S UTC")
            if raw_start_time != ann_start_time:
                continue
            try:
                ann = read_annotations(self.label_path(subject)[0][0])
            except Exception as e:
                ann = read_annotations(self.label_path_url(subject)[0][0])
            remove_idx = []
            labels = []
            label_idx = []
            for j in range(len(ann.description)):
                onset_sec = ann.onset[j]
                duration_sec = ann.duration[j]
                ann_str = ann.description[j]
                label = self._EVENTS[ann_str]

                if label != 5:
                    if duration_sec % EPOCH_SEC_SIZE != 0:
                        raise Exception("Something wrong")

                    duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                    label_epoch = np.ones(duration_epoch, dtype=int) * label
                    labels.append(label_epoch)
                    idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                    label_idx.append(idx)

                else:
                    idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                    remove_idx.append(idx)

            labels = np.hstack(labels)

            if len(remove_idx) > 0:
                remove_idx = np.hstack(remove_idx)
                select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
            else:
                select_idx = np.arange(len(raw_ch_df))

            label_idx = np.hstack(label_idx)
            select_idx = np.intersect1d(select_idx, label_idx)
            if len(label_idx) > len(select_idx):
                n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                if n_trims == 0:
                    print("Labels have much more data than rawdata")
                    continue
                if n_label_trims != 1:
                    print("Deleted tag data exceeds 1")
                    print("n_trims: {}".format(n_trims))
                    print("n_label_trims: {}".format(n_label_trims))

                select_idx = select_idx[:-n_trims]
                labels = labels[:-n_label_trims]

            raw_ch = raw_ch_df.values[select_idx]

            if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("Raw data not divisible by 30S")

            n_epochs = int(len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate))

            x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
            y = labels.astype(np.int32)

            assert len(x) == len(y)
            try:
                filename = ntpath.basename(self.data_path(subject)[0][0]).replace("-PSG.edf", ".npz")
            except Exception as e:
                a = self.data_path_url(subject)[0][0]
                filename = ntpath.basename(a).replace("-PSG.edf", ".npz")
            save_dict = {
                "x": x,
                "y": y,
                "fs": sampling_rate,
                "ch_label": select_ch,
            }
            np.savez(os.path.join(update_path, filename), **save_dict)

    def get_processed_data(self,
                           subjects: List[Union[int, str]] = None,
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
            num_npz = self.count_npz_files(update_path)
            subjects = list(range(num_npz))
            print(f'subjects:{num_npz}')

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

        labels = np.array(labels)
        read_datas = read_datas.transpose(0, 2, 1)

        return [labels, read_datas]

    @staticmethod
    def count_npz_files(file):
        npz_files = glob.glob(os.path.join(file, '*.npz'))
        return len(npz_files)


if __name__ == "__main__":
    path = r'D:\sleep-data\ST\raw'         # 原始数据raw_data存储地址，没有则会自动下载
    dataPath = r'D:\sleep-data\ST\npz'     # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)

    subjects = [0, 1, 2]                   # None则代表处理所有被试
    select_ch = ["EEG Fpz-Cz"]             # None则代表使用单通道"EEG Fpz-Cz"
    num_classes = 5                        # 睡眠分期的分类任务，支持2-5类

    sleep = Sleep_telemetry(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete.")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print("labels.size: " + str(labels.size))
    print("datas.shape: " + str(read_datas.shape))
