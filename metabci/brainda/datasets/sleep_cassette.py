import os
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


class SleepCassette(Sleep_telemetry):

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
    path = r'D:\sleep-data\ST'
    dataPath = r'D:\sleep-data\ST\MNE-sleep_edf-data\files\sleep-edfx\1.0.0\sleep-cassette'
    sleep = SleepCassette(dataPath=path)
    sleep.save_processed_data(update_path=dataPath)
    data = sleep.get_processed_data(subjects=[1], update_path=dataPath)
    labels, read_datas = data[0], data[1]
    print(read_datas)
