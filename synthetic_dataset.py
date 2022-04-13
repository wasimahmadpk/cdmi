import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from riverdata import RiverData
import parameters
import netCDF
import math


class SyntheticDataset:

    def __init__(self, root, time_steps, Tref, C, Tao, ey, ez, er):

        self.time_steps = time_steps

        self.root = root
        self.C = C
        self.Tao = Tao
        self.ey = ey
        self.ez = ez
        self.er = er
        self.X1 = list(np.zeros(15))
        self.X2 = list(np.zeros(15))
        self.X3 = list(np.zeros(15))
        self.X4 = list(np.zeros(15))
        self.X5 = list(np.zeros(15))
        self.X6 = list(np.zeros(15))
        self.X7 = list(np.zeros(15))
        self.X8 = list(np.zeros(15))
        self.X9 = list(np.zeros(15))
        self.X10 = list(np.zeros(15))

    def normalize(self, var):
        nvar = (np.array(var) - np.mean(var)) / np.std(var)
        return nvar

    def down_sample(self, data, win_size):
        agg_data = []
        monthly_data = []
        for i in range(len(data)):
            monthly_data.append(data[i])
            if (i % win_size) == 0:
                agg_data.append(sum(monthly_data) / win_size)
                monthly_data = []
        return agg_data

    def generate_data(self):

        for t in range(10, self.time_steps):

            self.X1.append(self.root[t])
            self.X2.append(C.get('c2') * self.X1[t - Tao.get('t1')] + ey[t])
            self.X3.append(C.get('c1') ** ((self.X1[t - Tao.get('t2')]) / 2 + ez[t]))
            self.X4.append(C.get('c3') * self.X1[t - Tao.get('t4')] + er[t])
            self.X5.append(C.get('c5') * self.X2[t - Tao.get('t1')] + self.X3[t - Tao.get('t2')] + self.X4[t - Tao.get('t3')] + ey[t])
        return self.X1, self.X2, self.X3, self.X4, self.X5

    def SNR(self, s, n):

        Ps = np.sqrt(np.mean(np.array(s)**2))
        Pn = np.sqrt(np.mean(np.array(n)**2))
        SNR = Ps/Pn
        return 10*math.log(SNR, 10)        # 10*math.log(((Ps-Pn)/Pn), 10)


if __name__ == '__main__':

    def generate_sine_wave(freq, sample_rate, duration):
        t = np.linspace(0, duration, sample_rate * duration, endpoint=False)
        frequencies = t * freq
        # 2pi because np.sin takes radians
        y = np.sin((2 * np.pi) * frequencies)
        return t, y

    # Generate sine wave
    pars = parameters.get_sig_params()
    SAMPLE_RATE = pars.get("sample_rate")  # Hertz
    DURATION = pars.get("duration")  # Seconds

    # Generate a 2 hertz sine wave that lasts for 5 seconds
    # t, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)

    _, nice_wave = generate_sine_wave(400, SAMPLE_RATE, DURATION)
    _, noise_wave = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
    noise_wave = noise_wave * 0.3
    noise = np.random.normal(0, 1, len(nice_wave))
    root = nice_wave + noise_wave + noise

    time_steps, Tref = round(len(root)), 15
    ey = np.random.normal(0, 0.05, time_steps)
    ez = np.random.normal(0, 0.15, time_steps)
    er = np.random.normal(0, 0.10, time_steps)

    C = {'c1': 0.95, 'c2': 1.5, 'c3': 2.50, 'c4': 0.75, 'c5': 0.99}          # c2:1.75, c5:1.85
    Tao = {'t1': 2, 't2': 1, 't3': 4, 't4': 3, 't5': 5, 't6': 6}
    data_obj = SyntheticDataset(root, time_steps, Tref, C, Tao, ey, ez, er)
    X1, X2, X3, X4, X5 = data_obj.generate_data()

    corr1 = np.corrcoef(ey, ez)

    print("Correlation Coefficient (ey, ez): ", corr1)
    # print("SNR (Temperature)", data_obj.SNR(Yts, ez))

    data = {'Z1': X1[150:], 'Z2': X2[150:], 'Z3': X3[150:], 'Z4': X4[150:], 'Z5': X5[150:]}
    df = pd.DataFrame(data, columns=['Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    # df.to_csv(r'/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/synthetic_data.csv', index_label=False, header=True)
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCausality/datasets/ncdata/synthetic_data.csv', index_label=False, header=True)
    print(df.head(100))

    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax1.plot(X1[150:1500])
    ax1.set_ylabel('X1')

    ax2 = fig.add_subplot(512)
    ax2.plot(X2[150:1500])
    ax2.set_ylabel("X2")

    ax3 = fig.add_subplot(513)
    ax3.plot(X3[150:1500])
    ax3.set_ylabel("X3")

    ax4 = fig.add_subplot(514)
    ax4.plot(X4[150:1500])
    ax4.set_ylabel("X4")

    ax5 = fig.add_subplot(515)
    ax5.plot(X5[150:1500])
    ax5.set_ylabel("X5")

    plt.show()