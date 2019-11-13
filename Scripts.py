from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import arange

'''
low-pass filter
_____[====]_R1____R2_[====]________________________
|             |             |                     |
|             |             |                 RL ___
|             |             |                    | |
e(t)          |             |                    | |
|           ____ C1      ____ C2                 ---
|           ____          ____                    |
|            |              |                     |
|____________|______________|____________|________|
'''

'''

du1/dt  = 1/C1(1/

'''


def G_s(R1: float, R2: float, C1: float, C2: float, S: float):
    V_out = 1. / R1 * R2 * C1 * C2
    V_in = S ** 2 + S * (1 / (R1 * C1) + 1 / (R2 * C1) + 1 / (R2 * C2)) + 1 / (R1 * R2 * C2 * C1)
    return V_out / V_in

def main():
    # Data:
    R1 = 1
    C1 = 0.032
    R2 = 1
    C2 = 0.032
    RL = 1

    step=0.001
    time_vector = np.arange(start=0, stop=1, step=step, dtype=float)
    # voltage_input_vector = np.ones((len(time_vector), 1)) * 2
    f = 10
    value_vector = np.sin(time_vector * 2 * np.pi * f)

    # plt.plot(time_vector, voltage_input_vector)
    plt.plot(time_vector, value_vector)
    plt.xlabel('[s]')
    plt.ylabel('[V]')
    plt.show()


if __name__ == "__main__":
    main()
