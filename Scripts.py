from decimal import Decimal

# import control as control
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


def G_s(R1: float, R2: float, C1: float, C2: float, S: float):
    V_out = 1. / (R1 * R2 * C1 * C2)
    V_in = S ** 2 + S * (1 / (R1 * C1) + 1 / (R2 * C1) + 1 / (R2 * C2)) + 1 / (R1 * R2 * C2 * C1)
    return V_out / V_in


def main():
    # Data:
    R1 = 1 * 1000
    C1 = 0.032 * 1e6
    R2 = 1 * 1000
    C2 = 0.032 * 1e6
    RL = 1 * 1000

    time = 1
    step = 0.01
    time_vector = np.arange(start=0, stop=time, step=step, dtype=float)
    f = 10
    value_vector = np.sin(time_vector * 2 * np.pi * f)

    const = 0
    f = const
    transfer_function = G_s(R1, R2, C1, C2, f)
    e_t = 2
    voltage_input_vector = [e_t]*len(time_vector)

    output_vector = []
    u_1 = 0.
    u_2 = 0.
    for t in time_vector:
        idx = int(t / step)
        du1_dt = (1. / C1) * ((1. / R1) * (e_t - u_1) - (1. / R2) * (u_1 - u_2))
        du2_dt = (1. / C2) * ((1. / R2) * (u_1 - u_2) - (u_2 / 1.e30))
        u_1 = du1_dt * t
        u_2 = du2_dt * t
        output_vector.append(u_2 * e_t)
        print("[{:5.5}: {:6.5};{:6.5} => {}]".format(t, u_1, u_2, output_vector[idx]))
        # print(output_vector)

    plt.plot(time_vector, voltage_input_vector, label='Input Voltage')
    # plt.plot(time_vector, value_vector)
    plt.plot(time_vector, np.array(output_vector), label='Output Voltage')
    plt.ylim(bottom=0, top=e_t*1.1)
    plt.xlabel('[s]')
    plt.ylabel('[V]')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
