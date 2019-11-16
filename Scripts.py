from decimal import Decimal

# import control as control
import matplotlib.pyplot as plt
import numpy as np
from numpy import transpose
from scipy import signal

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

'''
G(s)=\dfrac{V_{in}}{V_{out}}=
\dfrac{
    \dfrac{1}{
        R_1R_2C_1C_2
    }}{ s^2 + 
        s\dfrac{1}{R_1C_1} + 
        \dfrac{1}{R_2C_1} +
        \dfrac{1}{R_2C_2} + 
        \dfrac{1}{R_1R_2C_2C_1}}
'''


def G_s(R1: float, R2: float, C1: float, C2: float, step: float = None):
    V_out_ss = 0.
    V_out_s = 0.
    V_out_c = 1. / (R1 * R2 * C1 * C2)

    V_in_ss = 1.
    V_in_s = 1. / (R1 * C1) + 1. / (R2 * C1) + 1. / (R2 * C2)
    V_in_c = 1. / (R1 * R2 * C1 * C2)
    return signal.TransferFunction(
        [V_out_ss, V_out_s, V_out_c],
        [V_in_ss, V_in_s, V_in_c])


def main():
    # Data:
    R1 = 1 * 1000
    C1 = 0.032 / 1e6
    R2 = 1 * 1000
    C2 = 0.032 / 1e6
    # RL = 1 * 1000

    time = 0.001
    time = 0.001
    step = 0.00001
    e_t = 2.
    time_vector = np.arange(start=0, stop=time, step=step, dtype=float)
    input_const_vec = np.ones_like(time_vector) * e_t
    input_f_vec = np.sin(time_vector*2*np.pi*2000)
    # input_const_vec = input_f_vec
    input_vec = input_f_vec
    system = G_s(R1, R2, C1, C2, None)
    t_out, y_out_vec, x_out_vec = signal.lsim(system, input_vec, time_vector)
    # t_out, y_out_vec, x_out_vec = signal.lsim(system, input_const_vec, time_vector)
    # t_out, y_out_vec, x_out_vec = signal.lsim(system, input_const_vec, time_vector)

    plt.figure(0)
    plt.plot(time_vector, input_vec, label='Input f=0')
    plt.plot(time_vector, y_out_vec, label='Response f=0')

    plt.ylim(bottom=-2., top=e_t * 1.1)
    plt.xlabel('[s]')
    plt.ylabel('[V]')
    plt.legend()
    plt.grid()
    plt.show()

    # w, mag, phase = signal.bode(system)
    # plt.figure(1)
    # plt.plot(w, phase)

    plt.show()


if __name__ == "__main__":
    main()
