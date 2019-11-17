# import control as control
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


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



def plot_results(input_vec, t_out, y_out_vec):
    plt.figure(0)
    plt.plot(t_out, input_vec, label='Input f=0')
    plt.plot(t_out, y_out_vec, label='Response f=0')
    plt.ylim(bottom=np.min(input_vec) * 0.9, top=np.max(input_vec) * 1.1)
    plt.xlabel('[s]')
    plt.ylabel('[V]')
    plt.legend()
    plt.grid()
    plt.show()


def simulate_rlc_response(C1, C2, R1, R2, input_vec, time_vector):
    system = G_s(R1, R2, C1, C2, None)
    t_out, y_out_vec, x_out_vec = signal.lsim(system, input_vec, time_vector)
    return t_out, y_out_vec


def generate_rlc_parameters():
    R1 = 1 * 1000
    C1 = 0.032e-6
    R2 = 1 * 1000
    C2 = 0.032e-6
    # RL = 1 * 1000
    return C1, C2, R1, R2


def generate_time_vec():
    time = 0.001
    time_vector = np.linspace(start=0, stop=time)
    return time_vector


def generate_input_vecs(time_vector):
    e_t = 2.
    input_const_vec = np.ones_like(time_vector) * e_t
    f = 50
    input_f_50_vec = np.sin(time_vector * 2 * np.pi * f)
    f = 600
    input_f_600_vec = np.sin(time_vector * 2 * np.pi * f)
    f = 1.75e3
    input_f_1_75k_vec = np.sin(time_vector * 2 * np.pi * f)
    f = 12e3
    input_f_12k_vec = np.sin(time_vector * 2 * np.pi * f)
    f = 21e3
    input_f_21k_vec = np.sin(time_vector * 2 * np.pi * f)
    input_vecs = [input_const_vec,
                  input_f_50_vec,
                  input_f_600_vec,
                  input_f_1_75k_vec,
                  input_f_12k_vec,
                  input_f_21k_vec]
    return input_vecs


def main():
    # Data:
    C1, C2, R1, R2 = generate_rlc_parameters()

    time_vector = generate_time_vec()

    input_vecs = generate_input_vecs(time_vector)

    for input_vec in input_vecs:
        t_out, y_out_vec = simulate_rlc_response(C1, C2, R1, R2, input_vec, time_vector)
        plot_results(input_vec, t_out, y_out_vec)


if __name__ == "__main__":
    main()
