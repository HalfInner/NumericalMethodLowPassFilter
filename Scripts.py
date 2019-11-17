# import control as control
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


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


class StaticFigureEnumerator:
    fig_num = 0


def plot_results(title, input_vec, frequency, t_out, y_out_vec):
    StaticFigureEnumerator.fig_num += 1
    plt.figure(StaticFigureEnumerator.fig_num)

    s_to_ms_scalar = 1e3
    plt.plot(t_out * s_to_ms_scalar, input_vec, label='Input f={}Hz'.format(frequency))
    plt.plot(t_out * s_to_ms_scalar, y_out_vec, label='Response')
    plt.ylim(bottom=-2.1, top=2.1)
    plt.xlabel('[ms]')
    plt.ylabel('[V]')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def simulate_rlc_response(C1, C2, R1, R2, RL, input_vec, time_vec):
    system = G_s(R1, R2, C1, C2, None)
    t_out, y_out_vec, x_out_vec = signal.lsim(system, input_vec, time_vec)
    return t_out, y_out_vec


def simulate_rlc_du1_response(C1, C2, R1, R2, RL, input_vec, time_vec):
    u1 = 1.
    u2 = 1.

    output_vec = np.zeros_like(time_vec)
    for t_step in time_vec:
        # test e = input_vec[0]
        e = 2.
        u1_tmp = (1. / C1) * (1. / R1 * (e - u1) - 1. / R2 * (u1 - u2))
        u2_tmp = (1. / C2) * (1. / R2 * (u1 - u2) - u2 / RL)
        u1, u2 = u1_tmp, u2_tmp
        output_vec[0] = u2 * e

    return time_vec, output_vec


def generate_rlc_parameters():
    R1 = 1e3
    C1 = 0.032e-6
    R2 = 1e3
    C2 = 0.032e-6
    RL = 1e3
    return C1, C2, R1, R2, RL


def generate_time_vec(f):
    time = 0.001
    if f is not 0:
        time = 3. / f

    step = 0.001
    num = 1. / step
    return np.linspace(start=0, stop=time, num=num)


def generate_input_frequency_vecs():
    period = 2 * np.pi

    # a)
    f = 0
    e_t = 2.
    time_vec = generate_time_vec(f)
    input_const_vec = (np.ones_like(time_vec) * e_t, f, time_vec)

    # b)
    f = 50
    time_vec = generate_time_vec(f)
    input_f_50_vec = (np.sin(time_vec * period * f), f, time_vec)

    # c)
    f = 600
    time_vec = generate_time_vec(f)
    input_f_600_vec = (np.sin(time_vec * period * f), f, time_vec)

    # d)
    f = 1.75e3
    time_vec = generate_time_vec(f)
    input_f_1_75k_vec = (np.sin(time_vec * period * f), f, time_vec)

    # e)
    f = 12e3
    time_vec = generate_time_vec(f)
    input_f_12k_vec = (np.sin(time_vec * period * f), f, time_vec)

    # f)
    f = 21e3
    time_vec = generate_time_vec(f)
    input_f_21k_vec = (np.sin(time_vec * period * f), f, time_vec)

    # g)
    square_period = 0.5e-3
    f = 1/square_period
    time_vec = generate_time_vec(f)
    input_f_square_4kHz_vec = (signal.square(time_vec * period * f), f, time_vec)
    input_vecs = [input_const_vec,
                  input_f_50_vec,
                  input_f_600_vec,
                  input_f_1_75k_vec,
                  input_f_12k_vec,
                  input_f_21k_vec,
                  input_f_square_4kHz_vec]
    return input_vecs


def main():
    C1, C2, R1, R2, RL = generate_rlc_parameters()
    for input_vec, frequency, time_vec in generate_input_frequency_vecs():
        t_out, y_out_vec = simulate_rlc_response(C1, C2, R1, R2, RL, input_vec, time_vec)
        plot_results('Simulated output', input_vec, frequency, t_out, y_out_vec)

        # t_out, y_out_vec = simulate_rlc_du1_response(C1, C2, R1, R2, RL, input_vec, time_vec)
        # plot_results('du1 output', input_vec, frequency, t_out, y_out_vec)

        # temporary
        # break


if __name__ == "__main__":
    main()
