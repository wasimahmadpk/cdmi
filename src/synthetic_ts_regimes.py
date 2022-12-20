# import netCDF
import math
import random
import parameters
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import preprocessing as prep
import matplotlib.pyplot as plt

# np.random.seed(1)

class SyntheticDataset:

    def __init__(self, roots, time_steps, Tref, C, Tao, ey, ez, er):

        self.time_steps = time_steps

        self.roots = roots
        self.C = C
        self.Tao = Tao
        self.ey = ey
        self.ez = ez
        self.er = er
        self.X1 = list(np.zeros(10))
        self.X2 = list(np.zeros(10))
        self.X3 = list(np.zeros(10))
        self.X4 = list(np.zeros(10))
        self.X5 = list(np.zeros(10))

    def generate_data(self):

        for t in range(11, self.time_steps):

            if t < 1000:
                self.X1.append(self.roots[0][t])
                self.X2.append(C.get('c1') * self.X1[t - Tao.get('t1')] + ey[t])
                self.X3.append(C.get('c2') ** ((self.X1[t - Tao.get('t2')]) / 2) + ez[t])
                self.X4.append(C.get('c3') * self.X1[t - Tao.get('t3')] + er[t])
                self.X5.append(C.get('c5') * self.X2[t - Tao.get('t4')] + C.get('c3') * self.X3[t - Tao.get('t4')] + C.get('c5') * self.X4[t - Tao.get('t4')]  + ey[t])
            elif t >= 1000 and t < 2000:
                self.X1.append(self.roots[1][t])
                self.X2.append(C.get('c1') * self.X1[t - Tao.get('t1')] + ey[t])
                self.X3.append(C.get('c2') ** ((self.X1[t - Tao.get('t2')]) / 2) + ez[t])
                self.X4.append(C.get('c3') * self.X2[t - Tao.get('t3')] + er[t])
                self.X5.append(C.get('c5') * self.X2[t - Tao.get('t4')] + C.get('c5') * self.X3[t - Tao.get('t4')] + ey[t])
            elif t >= 2000:
                self.X1.append(self.roots[2][t])
                self.X2.append(C.get('c1') * self.X1[t - Tao.get('t1')] + ey[t])
                self.X3.append(C.get('c2') ** ((self.X1[t - Tao.get('t2')]) / 2) + ez[t])
                self.X4.append(C.get('c3') * self.X2[t - Tao.get('t3')] + er[t])
                self.X5.append(C.get('c5') * self.X3[t - Tao.get('t4')] + ey[t])

        return self.X1, self.X2, self.X3, self.X4, self.X5


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

    roots = []
    eys, ezs, ers = [], [], []
    
    for r in range(3):
        _, nice_wave = generate_sine_wave(400, SAMPLE_RATE, DURATION)
        _, noise_wave = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
        noise_wave = noise_wave * 0.30
        noise = np.random.normal(r, (r+1)*1.0, len(nice_wave))
        root = nice_wave + noise_wave + noise

        roots.append(root)

    # root = np.random.normal(0, 1.0, 2000)
    time_steps, Tref = 3110, 15
    # np.random.seed(1)
    ey = np.random.normal(0, 0.35, time_steps)
    # np.random.seed(2)
    ez = np.random.normal(0, 0.75, time_steps)
    # np.random.seed(3)
    er = np.random.normal(0, 0.30, time_steps)

    C = {'c1': 0.95, 'c2': 1.50, 'c3': 0.85, 'c4': 1.00, 'c5': 0.99}           # c2:1.75, c5:1.85
    Tao = {'t1': 2 , 't2': 2, 't3': 2, 't4': 3, 't5': 5, 't6': 6}
    data_obj = SyntheticDataset(roots, time_steps, Tref, C, Tao, ey, ez, er)
    X1, X2, X3, X4, X5 = data_obj.generate_data()

    # print("SNR (Temperature)", data_obj.SNR(Yts, ez))

    data = {'Z1': X1[101:], 'Z2': X2[101:], 'Z3': X3[101:], 'Z4': X4[101:], 'Z5': X5[101:]}
    df = pd.DataFrame(data, columns=['Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCausality/datasets/synthetic_datasets/synthetic_data_regimes.csv', index_label=False, header=True)
    print(df.head(100))
    print("Correlation Matrix")
    print(df.corr(method='pearson'))

    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax1.plot(X1[150:])
    ax1.set_ylabel('X1')

    ax2 = fig.add_subplot(512)
    ax2.plot(X2[150:])
    ax2.set_ylabel("X2")

    ax3 = fig.add_subplot(513)
    ax3.plot(X3[150:])
    ax3.set_ylabel("X3")

    ax4 = fig.add_subplot(514)
    ax4.plot(X4[150:])
    ax4.set_ylabel("X4")
    
    ax5 = fig.add_subplot(515)
    ax5.plot(X5[150:])
    ax5.set_ylabel("X5")
    plt.show()