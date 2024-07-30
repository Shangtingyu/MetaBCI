import os
from datetime import datetime

import numpy as np
from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.AttnSleep import AttnSleep
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


def save_res_pre(X: np.ndarray, parampath: str, train_selection=AttnSleep) -> np.ndarray:
    """
    Predicts sleep stages using a pre-trained AttnSleep model and saves the true and predicted labels.

    Parameters:
    X (np.ndarray): The input data.
    parampath (str): The file path to the pre-trained model parameters.
    train_selection: The function to initialize the AttnSleep model.

    Returns:
    np.ndarray: The predicted labels.
    """
    # Initialize the AttnSleep model with 2 input channels and 5 output classes
    model = train_selection(2, 5)
    model.initialize()

    # Load the model parameters from the specified path
    model.load_params(f_params=parampath)

    # Convert the input data and labels to tensor format and adjust dimensions
    x = np_to_th(X)

    # Predict sleep stages using the model
    y_predict = model.predict(x)

    # Get current time and format the filename
    current_time = datetime.now().strftime("%m%d%H%M%S")
    filename = f"predict{current_time}.npy"

    # Create the pre_res directory if it doesn't exist
    save_dir = os.path.join(os.getcwd(), 'pre_res')
    os.makedirs(save_dir, exist_ok=True)

    # Save the predicted labels to a file in the pre_res directory
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, y_predict)
    print(f'Save results to {save_path}')

    return y_predict


def main() -> None:
    # Define the paths to the data and model parameters
    datapath = r'D:\shhs\EEG_EOG'
    parampath = r'checkpoints\20240723_151225\params.pt'
    sleep_data = Sleep_telemetry()
    datas = sleep_data.get_processed_data(subjects=[10], update_path=datapath)
    data = datas[1]
    # Call the function to perform prediction and save results
    save_res_pre(data, parampath)


if __name__ == '__main__':
    main()
