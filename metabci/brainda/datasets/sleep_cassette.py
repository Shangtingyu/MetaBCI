import os
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


class Sleep_cassette(Sleep_telemetry):
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
        B Kemp, AH Zwinderman, B Tuk, HAC Kamphuisen, JJL Oberyé. Analysis of a
        sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.
        IEEE-BME 47(9):1185-1194 (2000).
        https://physionet.org/content/sleep-edfx/1.0.0/

        The sleep-edf database contains 197 whole-night polygraphic sleep recordings, which
        include electroencephalography (EEG), electrooculography (EOG), electromyography
        (EMG) of the chin, and event markers. Some records also contain respiration and body
        temperature. The corresponding hypnograms (sleep patterns) were manually scored by
        well-trained technicians according to the Rechtschaffen and Kales manual, and are
        also available. The data originates from two studies, which are briefly described below.

        The SC (Sleep Cassette) was obtained in a 1987-1991 study of age effects on sleep in healthy
        Caucasians aged 25-101, without any sleep-related medication [2]. Two PSGs of approximately
        20 hours each were recorded during two subsequent day-night periods at the subjects' homes.
    """
    def __init__(self, dataPath: str = None):
        super().__init__(
            dataPath=dataPath
        )
        self.dataPath = dataPath
        self.update_path = None
        self.subjects = list(range(152))
        self.sleep_URL = 'https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'

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
        # Get the absolute path to the current file
        current_file = os.path.abspath(__file__)
        project_root = current_file

        # Find the project root directory
        while not os.path.isfile(os.path.join(project_root, 'setup.py')):
            parent_dir = os.path.dirname(project_root)
            if parent_dir == project_root:  # Reached the root directory
                raise FileNotFoundError("setup.py not found in any parent directory")
            project_root = parent_dir

        hypnogram_file_path = os.path.join(project_root, 'docs', 'cassette_Hypnogram.txt')
        psg_file_path = os.path.join(project_root, 'docs', 'cassette_psg.txt')

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


if __name__ == "__main__":
    path = r'D:\sleep-data\SC'                                      # 原始数据raw_data存储地址，没有则会自动下载
    dataPath = r'D:\sleep-data\SC'                                  # 数据预处理后的npz_data存储地址
    subjects = [0, 1, 2]                                            # None则代表处理所有被试
    select_ch = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]       # None则代表使用单通道"EEG Fpz-Cz"
    num_classes = 3                                                 # 睡眠分期的分类任务，支持2-5类

    sleep = Sleep_cassette(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete and data loading begins")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print("labels.size: " + str(labels.size))
    print("datas.shape: " + str(read_datas.shape))
