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

    def __init__(self, root, time_steps, Tref, C, Tao, ex, ey, ez):

        self.time_steps = time_steps

        self.root = root
        self.C = C
        self.Tao = Tao
        self.ex = ex
        self.ey = ey
        self.ez = ez
        self.X1 = list(np.zeros(10))
        self.X2 = list(np.zeros(10))
        self.X3 = list(np.zeros(10))
        self.X4 = list(np.zeros(10))
        self.X5 = list(np.zeros(10))

    def generate_data(self):

        for t in range(10, self.time_steps):

            if t < self.time_steps/3:
                ex = self.ex[0]
                ey = self.ey[0]
                ez = self.ez[0]

            if t >= self.time_steps/3 and t < 2*self.time_steps/3:
                ex = self.ex[1]
                ey = self.ey[1]
                ez = self.ez[1]

            elif t >= 2 * self.time_steps/3:
                ex = self.ex[2]
                ey = self.ey[2]
                ez = self.ez[2]

            self.X1.append(self.root[t] + ex[t])
            self.X2.append(C.get('c1') * self.X1[t - Tao.get('t1')] + ey[t])
            self.X3.append(C.get('c2') ** ((self.X1[t - Tao.get('t2')])/2) + ez[t])
            self.X4.append(C.get('c3') ** (self.X3[t - Tao.get('t2')]/2) + ex[t])
            self.X5.append(C.get('c1') * self.X1[t - Tao.get('t3')] + C.get('c4') * self.X4[t - Tao.get('t1')] + ey[t])
        
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

    _, nice_wave = generate_sine_wave(400, SAMPLE_RATE, DURATION)
    _, noise_wave = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
    noise_wave = noise_wave * 0.50
    noise = np.random.normal(2, 1.10, len(nice_wave))
    root = noise

    # root = np.random.normal(0, 1.0, 2000)
    time_steps, Tref = 1200, 15

    ex, ey, ez = [], [], []
    for i in range(3):
        ex.append(np.random.normal(i + i*0.1, 0.30, time_steps))
        ey.append(np.random.normal(i + i*0.1, 0.20, time_steps))
        ez.append(np.random.normal(i + i*0.1, 0.25, time_steps))

    C = {'c1': 0.70, 'c2': 1.50, 'c3': 0.75, 'c4': 1.25, 'c5': 0.90}           # c2:1.75, c5:1.85
    Tao = {'t1': 2, 't2': 3, 't3': 4, 't4': 1, 't5': 6, 't6': 5}
    data_obj = SyntheticDataset(root, time_steps, Tref, C, Tao, ex, ey, ez)
    X1, X2, X3, X4, X5 = data_obj.generate_data()

    data = {'Z1': X1[50:], 'Z2': X2[50:], 'Z3': X3[50:], 'Z4': X4[50:], 'Z5': X5[50:]}
    df = pd.DataFrame(data, columns=['Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
    df.to_csv(r'/home/ahmad/PycharmProjects/deepCausality/datasets/synthetic_datasets/synthetic_datac.csv', index_label=False, header=True)
    print(df.head(10))
    print("Correlation Matrix")
    print(df.corr(method='pearson'))

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
    #
    ax5 = fig.add_subplot(515)
    ax5.plot(X5[150:1500])
    ax5.set_ylabel("X5")

    plt.show()