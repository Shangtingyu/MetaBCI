import os
import glob


def save_edf_filenames_to_txt(edf_folder, txt_filename):
    """
    Save the names of all .edf files in a folder to a .txt file, one filename per line.

    Parameters
    ----------
    edf_folder : str
        The path to the folder containing .edf files.
    txt_filename : str
        The name of the .txt file where filenames will be saved.
    """
    # Get a list of all .edf files in the folder
    edf_files = glob.glob(os.path.join(edf_folder, "*Hypnogram.edf"))

    # Get the current working directory
    current_directory = os.getcwd()
    # Construct the full path for the .txt file
    txt_file = os.path.join(current_directory, txt_filename)

    with open(txt_file, 'w') as f:
        for edf_file in edf_files:
            # Write each .edf file name to the text file
            f.write(os.path.basename(edf_file) + '\n')

    print(f"Saved filenames to {txt_file}")


# Example usage
edf_folder = r"C:\Users\86130\Desktop\metabci\01_SleepEDF\sleep-cassette"
txt_file = 'cassette_Hypnogram.txt'
save_edf_filenames_to_txt(edf_folder, txt_file)
