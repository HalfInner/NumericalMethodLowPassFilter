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


def plot_results(title, input_vec, frequency, t_out, y_out_vecs: list):
    StaticFigureEnumerator.fig_num += 1
    plt.figure(StaticFigureEnumerator.fig_num / 2)

    s_to_ms_scalar = 1e3
    plt.plot(t_out * s_to_ms_scalar, input_vec, label='Input f={}Hz'.format(frequency))

    idx = 0
    for y_out_vec in y_out_vecs:
        idx += 1
        plt.plot(t_out * s_to_ms_scalar, y_out_vec, label='Response du{}'.format(idx))

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


def calculate_du1(C1, C2, R1, R2, RL, e_n, u1_n, u2_n):
    out = (1. / C1) * (1. / R1 * (e_n - u1_n) - 1. / R2 * (u1_n - u2_n))
    return out


def calculate_du2(C1, C2, R1, R2, RL, e_n, u1_n, u2_n):
    out = (1. / C2) * (1. / R2 * (u1_n - u2_n) - u2_n / RL)
    return out


def simulate_rlc_du1_response_euler(C1, C2, R1, R2, RL, input_vec, time_vec, frequency):
    u1_vec = np.zeros_like(time_vec)
    u2_vec = np.zeros_like(time_vec)

    step = (time_vec[-1] - time_vec[0]) / len(time_vec)
    for idx in range(len(time_vec) - 1):
        u1_tmp = calculate_du1(C1, C2, R1, R2, RL, input_vec[idx], u1_vec[idx], u2_vec[idx])
        u2_tmp = calculate_du2(C1, C2, R1, R2, RL, input_vec[idx], u1_vec[idx], u2_vec[idx])
        u1_vec[idx + 1] = u1_vec[idx] + step * u1_tmp
        u2_vec[idx + 1] = u2_vec[idx] + step * u2_tmp

    return time_vec, u1_vec, u2_vec


def simulate_rlc_du1_response_extended_euler(C1, C2, R1, R2, RL, input_vec, time_vec, frequency):
    u1_vec = np.zeros_like(time_vec)
    u2_vec = np.zeros_like(time_vec)

    step = (time_vec[-1] - time_vec[0]) / len(time_vec)
    for idx in range(len(time_vec) - 1):
        u1_tmp = calculate_du1(C1, C2, R1, R2, RL, input_vec[idx], u1_vec[idx], u2_vec[idx])
        u2_tmp = calculate_du2(C1, C2, R1, R2, RL, input_vec[idx], u1_vec[idx], u2_vec[idx])

        half_input = input_vec[idx] + step / 2.
        half_u1 = u1_vec[idx] + step / 2 * u1_tmp
        half_u2 = u2_vec[idx] + step / 2 * u2_tmp

        main_u1_tmp = calculate_du1(C1, C2, R1, R2, RL, half_input, half_u1, half_u2)
        main_u2_tmp = calculate_du2(C1, C2, R1, R2, RL, half_input, half_u1, half_u2)

        u1_vec[idx + 1] = u1_vec[idx] + step * main_u1_tmp
        u2_vec[idx + 1] = u2_vec[idx] + step * main_u2_tmp

    return time_vec, u1_vec, u2_vec


def generate_rlc_parameters():
    R1 = 1e3
    C1 = 0.032e-6
    R2 = 1e3
    C2 = 0.032e-6
    RL = 1e30
    # RL = 1e3
    return C1, C2, R1, R2, RL


def generate_time_vec(f):
    time = 0.001
    if f is not 0:
        # TODO(KB): multiplying time cause some errors in Euler Method
        time = 1. / f

    step = 0.01
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
    f = 50.
    e_t = 1.
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
    f = 1 / square_period
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
        plot_results('Simulated output', input_vec, frequency, t_out, [y_out_vec])

        t_out, du1_out_vec, du2_out_vec = simulate_rlc_du1_response_euler(
            C1, C2, R1, R2, RL, input_vec, time_vec, frequency)
        plot_results('euler', input_vec, frequency, t_out, [du1_out_vec, du2_out_vec])

        t_out, du1_out_vec, du2_out_vec = simulate_rlc_du1_response_extended_euler(
            C1, C2, R1, R2, RL, input_vec, time_vec, frequency)
        plot_results('extended euler', input_vec, frequency, t_out, [du1_out_vec, du2_out_vec])

        # temporary
        # break


if __name__ == "__main__":
    main()
