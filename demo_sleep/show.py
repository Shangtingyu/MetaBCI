import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from demo_sleep.predict import save_res_pre, save_pre_score
from demo_sleep.smooth import smooth
from metabci.brainda.datasets.sleep_telemetry import Sleep_telemetry
from metabci.brainda.utils.performance import Performance, _confusion_matrix

# plt.rcParams是一个全局配置对象,设置显示中文字体和负号
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (10, 8)


def plotAnalyze(data: np.ndarray) -> None:
    """
    Analyze the distribution of sleep stages and plot a pie chart.

    Parameters:
    data (np.ndarray): Array of sleep stage labels.

    Returns:
    None
    """

    def make_autopct(values):
        """
        Generate a function to format the pie chart with both percentages and values.

        Parameters:
        values (list): List of values to display in the pie chart.

        Returns:
        function: A function to format the pie chart labels.
        """
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            # Display both the percentage and the actual value
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)

        return my_autopct

    print(f"Length of data: {len(data)}")
    num_classes = len(np.unique(data))
    data = data.tolist()
    data_count = [data.count(i) for i in range(num_classes)]

    print(f"Number of epochs in Wake stage: {data_count[0]}")
    print(f"Number of epochs in N1 stage: {data_count[1]}")
    print(f"Number of epochs in N2 stage: {data_count[2]}")
    print(f"Number of epochs in N3 stage: {data_count[3]}")
    print(f"Number of epochs in REM stage: {data_count[4]}")

    # Sizes for each sleep stage
    sizes = data_count
    # Labels for each sleep stage
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    # Colors for each sleep stage in the pie chart
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen']

    # Create the pie chart
    plt.figure(1)
    plt.pie(sizes, labels=labels, colors=colors, autopct=make_autopct(sizes))

    # Title of the pie chart
    plt.title("MDSK Sleep Stage Distribution")

    # Display the pie chart
    plt.show()
    plt.clf()


def plotTime(ax, data: np.ndarray, flag_modi: bool = False, color: str = "blue", name: str = "") -> None:
    """
    Plot the model's output results as a sleep trend chart.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot the data.
    data (np.ndarray): The model's predicted label values.
    flag_modi (bool): Whether to modify the data using specific rules. Defaults to False.
    color (str): The color of the plot. Defaults to "blue".
    name (str): The title of the plot. Defaults to an empty string.

    Returns:
    None
    """
    time_interval = 30  # Time interval represented by each label (in seconds)
    sleep_stages = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}  # Mapping numeric values to sleep stages

    if flag_modi:
        # Apply modifications to the data according to specific rules
        flag_modi_1 = flag_modi_2 = flag_modi_3 = flag_modi_4 = flag_modi_5 = flag_modi_6 = True

        if flag_modi_1:
            # Set the first 60 minutes to Wake
            time_wake = 55  # Time in minutes
            num_label = int(time_wake * 60 / time_interval)
            data[0:num_label] = 0

        if flag_modi_3:
            # Only consider REM stage if there are at least 10 consecutive REM epochs
            min_rem_epochs = 10
            for i in range(1, len(data) - min_rem_epochs + 1):
                if all(data[i + j] == 4 for j in range(min_rem_epochs)):
                    data[i:i + min_rem_epochs] = 4

        if flag_modi_4:
            # If the gap between two REM epochs is less than 5 Non-REM epochs, consider them all as REM
            max_non_rem_gap = 5
            for i in range(1, len(data) - 1):
                if data[i] == 4 and data[i - 1] == 4 and data[i + 1] != 4:
                    gap_count = 0
                    j = i + 1
                    while j < len(data) and data[j] != 4:
                        gap_count += 1
                        j += 1
                    if gap_count <= max_non_rem_gap:
                        data[i + 1:j] = 4

        if flag_modi_5:
            # If an epoch is considered Wake, the next index_awake epochs are also considered Wake
            index_awake = 5
            i = 0
            while i < len(data):
                if data[i] == 0:
                    data[i + 1:i + 1 + index_awake] = [0] * index_awake
                    i += 1 + index_awake  # Skip these epochs
                else:
                    i += 1

        if flag_modi_2:
            # Changes in values should last for at least 3 epochs to be considered a change
            min_epochs_to_change = 3
            current_epoch_count = 0
            for i in range(1, len(data)):
                if data[i] != data[i - 1]:
                    current_epoch_count += 1
                    if current_epoch_count < min_epochs_to_change:
                        data[i] = data[i - 1]
                else:
                    current_epoch_count = 0

        if flag_modi_6:
            # After a random number of REM epochs, change the subsequent random number of epochs to Wake or N2
            rem_count = 0
            for i in range(len(data)):
                rem_count_threshold = random.randint(30, 150)
                wake_change_interval = random.randint(30, 60)
                n2_change_interval = random.randint(5, 49)
                flag = random.randint(0, 2)
                if data[i] == 4:  # If current epoch is REM
                    rem_count += 1
                    if rem_count >= rem_count_threshold:
                        if flag == 1:
                            # Change subsequent epochs to Wake
                            for j in range(1, wake_change_interval + 1):
                                if i + j < len(data):
                                    data[i + j] = 0
                        else:
                            # Change subsequent epochs to N2
                            for j in range(1, n2_change_interval + 1):
                                if i + j < len(data):
                                    data[i + j] = 2
                else:
                    rem_count = 0

    start_time = datetime.strptime("21:00", "%H:%M")

    time_axis = [start_time + timedelta(seconds=i * time_interval) for i in range(len(data))]

    for i in range(len(data) - 1):
        t1, t2 = time_axis[i], time_axis[i + 1]
        value1, value2 = data[i], data[i + 1]

        ax.hlines(value1, t1, t2, colors=color)

        if value1 != value2:
            ax.vlines(t2, min(value1, value2), max(value1, value2), colors=color)

    # Plot the horizontal line for the last data point
    ax.hlines(data[-1], time_axis[-1], time_axis[-1] + timedelta(seconds=time_interval), colors=color)

    ax.set_title(name, fontsize=14, fontweight='bold')

    # Modify tick labels
    ax.set_xlim(datetime.strptime("21:00", "%H:%M"), datetime.strptime("06:00", "%H:%M") + timedelta(days=1))
    ax.set_yticks(list(sleep_stages.keys()))
    ax.set_yticklabels(list(sleep_stages.values()))
    ax.invert_yaxis()  # Invert Y-axis

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.get_xticklabels(), ha='center')


def main():
    performance = Performance(estimators_list=["Acc", "TPR", "FNR", "TNR"], isdraw=True)
    datapath = r'D:\sleep-data\ST\EEG Fpz-Cz'  # 数据预处理后的npz_data存储地址
    parampath = r'D:\metabci\demo_sleep\checkpoints\20240802_100342\params.pt'  # 保存模型参数的地址
    sleep_data = Sleep_telemetry()
    subjects = [1]  # 对于作图只能有一个对象
    datas = sleep_data.get_processed_data(subjects=subjects, update_path=datapath)
    y_predict = save_res_pre(datas[1], parampath)
    y_true = datas[0]
    y_predict_sm = smooth(y_predict)
    print(performance.evaluate(y_true, y_predict))  # 打印评估结果
    print(performance.evaluate(y_true, y_predict_sm))
    plotAnalyze(y_predict_sm)  # 绘制分期占比饼图
    fig, axs = plt.subplots(3, 1, figsize=(15, 10))
    plotTime(axs[0], y_true, flag_modi=False, color="black", name="PSG true label")
    plotTime(axs[1], y_predict, flag_modi=False, color="GoldenRod", name="prediction")
    plotTime(axs[2], y_predict_sm, flag_modi=False, color="GoldenRod", name="smooth prediction")
    plt.tight_layout()
    plt.show()  # 绘制分期趋势图


if __name__ == "__main__":
    main()
