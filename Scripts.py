from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import arange

'''
low-pass filter
_____[====]_R1____R2_[====]________________________
|             |             |                     |
|             |             |                 R3 ___
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
    milliseconds = 1000
    samples = np.arange(milliseconds)

    R1 = 1000.
    C1 = 0.032

    plt.plot(n, faza)
    plt.xlabel('sample')
    plt.ylabel('kąt fazowy [rad]')
    plt.title('Sygnał Fazowy')
    plt.show()


if __name__ == "__main__":
    main()
