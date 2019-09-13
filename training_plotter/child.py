import json
import sys
import threading
import traceback
from collections import deque
from typing import Sequence, Tuple, List, Deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QFont

TRAINING_LENGTH = 2000
EVAL_LENGTH = 40

training_i: Deque[int] = deque(maxlen=TRAINING_LENGTH)
training_duration: Deque[float] = deque(maxlen=TRAINING_LENGTH)
training_reward: Deque[float] = deque(maxlen=TRAINING_LENGTH)
training_duration_run_max: float = 0

eval_i: Deque[int] = deque(maxlen=EVAL_LENGTH)
eval_duration: Deque[float] = deque(maxlen=EVAL_LENGTH)
eval_reward: Deque[float] = deque(maxlen=EVAL_LENGTH)
eval_duration_run_max: float = 0

plots: List[Tuple[str, Sequence, Sequence, int]] = [
    # Label, data_x, data_y, average
    ('Training Duration', training_i, training_duration, 50),
    ('Evaluation Duration', eval_i, eval_duration, 10),
]


def read_input():
    try:
        while True:
            try:
                line = input()
            except EOFError:
                return
            message = json.loads(line)
            record_episode_to_graph(message['i'], message['d'], message['r'], message['e'])
            # print(message)
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(-1)


def record_episode_to_graph(i: int, duration: float, reward: float, evaluation: bool):
    if evaluation:
        eval_i.append(i)
        eval_duration.append(duration)
        eval_reward.append(reward)
        global eval_duration_run_max
        if duration > eval_duration_run_max:
            eval_duration_run_max = duration
    else:
        training_i.append(i)
        training_duration.append(duration)
        training_reward.append(reward)
        global training_duration_run_max
        if duration > training_duration_run_max:
            training_duration_run_max = duration


def rolling_mean_no_edges(a: np.ndarray, n: int = 3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def rolling_mean(x, n: int = 3):
    out = np.zeros_like(x, dtype=np.float32)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if n % 2 == 0:
            a, b = i - (n - 1) // 2, i + (n - 1) // 2 + 2
        else:
            a, b = i - (n - 1) // 2, i + (n - 1) // 2 + 1

        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


if __name__ == '__main__':
    # pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'w')

    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget()
    win.setWindowTitle('Eagle')
    win.setGeometry(0, 30, 450, 600)
    win.setAttribute(98)  # Qt::WA_ShowWithoutActivating
    win.show()

    threading.Thread(target=read_input, daemon=True).start()

    plot_items = []
    plot_data_items = []
    label_items = []
    for i, (label, data_x, data_y, average_len) in enumerate(plots):
        plot_item = win.addPlot(row=i, col=0, title=label)
        # plot_item.setLabel('bottom', label)

        right_axis = plot_item.getAxis('right')
        right_axis.show()
        right_axis.setStyle(showValues=False, autoExpandTextSpace=False)
        top_axis = plot_item.getAxis('top')
        top_axis.show()
        top_axis.setStyle(showValues=False, autoExpandTextSpace=False)

        # plot_item.showGrid(True, True, 0.3)
        plot_items.append(plot_item)
        plot_data_item_0 = plot_item.plot(pen=pg.mkPen(color='FFF7', width=3), antialias=True)
        plot_data_item_1 = plot_item.plot(pen=pg.mkPen(color='FFF3', width=1), antialias=True)
        plot_data_items.append((plot_data_item_0, plot_data_item_1))

        label_item = pg.LabelItem()
        # label_item.item.setDefaultTextColor(QColor("FFFF"))
        q_font = QFont("Consolas")
        q_font.setPointSize(7)
        label_item.item.setFont(q_font)
        label_items.append(label_item)
        win.addItem(label_item, row=i, col=1)


    def get_text(data_array: np.ndarray, window: int):
        if len(data_array) > 0:
            data_array_max = data_array.max()
        else:
            data_array_max = 0

        windowed_array = data_array[-2 * window:]
        if len(windowed_array) > 0:
            windowed_array_min = windowed_array.min()
            windowed_array_max = windowed_array.max()
            windowed_array_mean = windowed_array.mean()
            windowed_array_median = np.median(windowed_array)
        else:
            windowed_array_min = 0
            windowed_array_max = 0
            windowed_array_mean = 0
            windowed_array_median = 0
        text = (f"Run max: {data_array_max :.2f} <br><br>" +
                f"Min:     {windowed_array_min:.2f} <br>" +
                f"Max:     {windowed_array_max:.2f} <br>" +
                f"Median:  {windowed_array_median:.2f} <br>" +
                f"Mean:    {windowed_array_mean :.2f}").replace(" ", "&nbsp;")
        return text


    def redraw():
        global plot_items
        try:
            for i, (label, data_x, data_y, window) in enumerate(plots):
                plot_data_item_heavy, plot_data_item_light = plot_data_items[i]
                data_x_ = np.array(data_x)
                data_array = np.array(data_y)
                plot_data_item_heavy.setData(data_x_, rolling_mean(data_array, window))
                plot_data_item_light.setData(data_x_, data_array)

                label_item = label_items[i]
                label_item.setText(get_text(data_array, window))
        except Exception as e:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()


    timer = QtCore.QTimer()
    timer.timeout.connect(redraw)
    timer.start(500)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
