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
        B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, “Analysis of a sleep-dependent
        neuronal feedback loop: the slow-wave microcontinuity of the EEG,” IEEE Trans. Biomed. Eng.,
        vol. 47, no. 9, pp. 1185–1194, Sep. 2000, doi: 10.1109/10.867928.

        The 153 SC* files (SC = Sleep Cassette) were obtained in a 1987-1991 study of age effects on sleep in healthy
        Caucasians aged 25-101, without any sleep-related medication. Two PSGs of about 20 hours each were recorded
        during two subsequent day-night periods at the subjects homes. Files are named in the form SC4ssNEO-PSG.edf
        where ss is the subject number, and N is the night. The first nights of subjects 36 and 52, and the second night
        of subject 13, were lost due to a failing cassette or laserdisk.The EOG and EEG signals were each sampled at
        100 Hz. The submental-EMG signal was electronically highpass filtered, rectified and low-pass filtered after
        which the resulting EMG envelope expressed in uV rms (root-mean-square) was sampled at 1Hz. Oro-nasal airflow,
        rectal body temperature and the event marker were also sampled at 1Hz.
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
    path = r'D:\sleep-data\SC\raw'                                      # 原始数据raw_data存储地址，没有则会自动下载
    dataPath = r'D:\sleep-data\SC\npz'                                  # 数据预处理后的npz_data存储地址
    os.makedirs(dataPath, exist_ok=True)

    subjects = [0, 1, 2]                                                # None则代表处理所有被试
    select_ch = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]           # None则代表使用单通道"EEG Fpz-Cz"
    num_classes = 3                                                     # 睡眠分期的分类任务，支持2-5类

    sleep = Sleep_cassette(dataPath=path)
    sleep.save_processed_data(update_path=dataPath, subjects=subjects, select_ch=select_ch)
    print("Data preprocessing is complete.")
    data = sleep.get_processed_data(update_path=dataPath, subjects=subjects, num_classes=num_classes)
    labels, read_datas = data[0], data[1]
    print(read_datas)
