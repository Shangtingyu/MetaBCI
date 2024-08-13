import os
from datetime import datetime

import numpy as np
from metabci.brainda.algorithms.deep_learning import np_to_th
from metabci.brainda.algorithms.deep_learning.attnsleepnet import AttnSleepNet
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry


def save_res_pre(X: np.ndarray, parampath: str, train_selection=AttnSleepNet(1, 5)) -> np.ndarray:
    """
    Predicts sleep stages using a pre-trained AttnSleep model and saves the true and predicted labels.

    Parameters:
    X (np.ndarray): The input data.
    parampath (str): The file path to the pre-trained model parameters.
    train_selection: The function to initialize the AttnSleep model,like AttnSleepNet(1, 5).

    Returns:
    np.ndarray: The predicted labels.
    """

    model = train_selection
    model.initialize()
    model.load_params(f_params=parampath)
    x = np_to_th(X)
    y_predict = model.predict(x)
    current_time = datetime.now().strftime("%m%d%H%M%S")
    filename = f"predict{current_time}.npy"
    save_dir = os.path.join(os.getcwd(), 'pre_res')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, y_predict)
    print(f'Save results to {save_path}')

    return y_predict


def save_pre_score(X: np.ndarray, parampath: str, train_selection=AttnSleepNet(1, 5)) -> np.ndarray:
    """
    Predicts sleep stages using a pre-trained model and saves the true and predicted labels.

    Parameters:
    X (np.ndarray): The input data.
    parampath (str): The file path to the pre-trained model parameters.
    train_selection: The function to initialize the AttnSleep model.

    Returns:
    np.ndarray: The predicted labels.
    """

    model = train_selection
    model.initialize()
    model.load_params(f_params=parampath)
    x = np_to_th(X)
    y_score = model.predict_proba(x)
    current_time = datetime.now().strftime("%m%d%H%M%S")
    filename = f"predict_score{current_time}.npy"
    save_dir = os.path.join(os.getcwd(), 'pre_res')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, y_score)
    print(f'Save results to {save_path}')

    return y_score


def main() -> None:
    datapath = r'D:\sleep-data\ST\EEG Fpz-Cz'
    parampath = r'D:\metabci\demo_sleep\checkpoints\ST-Fpz-Cz\params.pt'
    subjects = [10]
    sleep_data = Sleep_telemetry()
    datas = sleep_data.get_processed_data(subjects=subjects, update_path=datapath)
    data = datas[1]
    save_res_pre(data, parampath)
    y_score = save_pre_score(data, parampath)
    print(y_score)


if __name__ == '__main__':
    main()
