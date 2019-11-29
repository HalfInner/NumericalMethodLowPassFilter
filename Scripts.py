# import control as control
import matplotlib.pyplot as plt
import numpy as np
import math

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
        if y_out_vec is None:
            continue
        plt.plot(t_out * s_to_ms_scalar, y_out_vec, label='Response du{}'.format(idx))

    plt.ylim(bottom=-2.1, top=2.1)
    plt.xlabel('[ms]')
    plt.ylabel('[V]')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig('{:02}_f{}.png'.format(StaticFigureEnumerator.fig_num, frequency), bbox_inches='tight')
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


def generate_time_vec(f, h, t=0.001):
    time = t
    if f is not 0:
        # TODO(KB): multiplying time cause some errors in Euler Method
        time = 1. / f

    step = h
    num = 1. / step
    return np.linspace(start=0, stop=time, num=num)


def generate_input_frequency_vecs(h=0.01, t=0.001):
    period = 2 * np.pi

    # test)
    f = 0
    e_t = 1.
    time_vec = generate_time_vec(f, h, t)
    # if abs(steps - h) > 0.001:
    #     raise Exception('Passed h={} is different from returned steps={}'.format(h, steps))
    input_const_test_vec = (np.ones_like(time_vec) * e_t, f, time_vec)

    # a)
    f = 0
    e_t = 2.
    time_vec = generate_time_vec(f, h)
    input_const_vec = (np.ones_like(time_vec) * e_t, f, time_vec)

    # b)
    f = 50.
    time_vec = generate_time_vec(f, h)
    input_f_50_vec = (np.sin(time_vec * period * f), f, time_vec)

    # c)
    f = 600
    time_vec = generate_time_vec(f, h)
    input_f_600_vec = (np.sin(time_vec * period * f), f, time_vec)

    # d)
    f = 1.75e3
    time_vec = generate_time_vec(f, h)
    input_f_1_75k_vec = (np.sin(time_vec * period * f), f, time_vec)

    # e)
    f = 12e3
    time_vec = generate_time_vec(f, h)
    input_f_12k_vec = (np.sin(time_vec * period * f), f, time_vec)

    # f)
    f = 21e3
    time_vec = generate_time_vec(f, h)
    input_f_21k_vec = (np.sin(time_vec * period * f), f, time_vec)

    # g)
    square_period = 0.5e-3
    f = 1 / square_period
    time_vec = generate_time_vec(f, h)
    input_f_square_4kHz_vec = (signal.square(time_vec * period * f) + 1, f, time_vec)

    input_vecs = [
        input_const_test_vec,
        input_const_vec,
        input_f_50_vec,
        input_f_600_vec,
        input_f_1_75k_vec,
        input_f_12k_vec,
        input_f_21k_vec,
        input_f_square_4kHz_vec
    ]
    return input_vecs


def simulate_damping_ration(C1, C2, R1, R2, RL):
    frequency_vec = np.arange(start=-1000, stop=3000, step=0.1)
    frequency_damping_vec = np.zeros_like(frequency_vec)

    for idx in range(len(frequency_vec)):
        freq = frequency_vec[idx]
        frequency_damping_vec[idx] = damping_value(C1, C2, R1, R2, RL, freq)

    return frequency_vec, frequency_damping_vec


def damping_value(C1, C2, R1, R2, RL, freq):
    damping_ration_const = 0.7017

    img_part = 1j * 2 * np.pi * freq

    equal_up_c = 1. / (R1 * R2 * C1 * C2)
    equal_down_s2 = img_part ** 2
    equal_down_s1 = img_part * (1. / (R1 * C1) + 1. / (R2 * C1) + 1. / (R2 * C2))
    equal_down_c = 1. / (R1 * R2 * C1 * C2)

    equal = equal_up_c / (equal_down_s2 + equal_down_s1 + equal_down_c)

    return damping_ration_const - np.abs(equal)


def find_fc_bisection(C1, C2, R1, R2, RL):
    freq_min = 0
    freq_max = 3e3

    epsilon = 0.0001
    bisection_freq = float('inf')
    bisection_gain = float('inf')
    while bisection_gain > epsilon:
        freq_avg = (freq_min + freq_max) / 2.
        gain_avg = damping_value(C1, C2, R1, R2, RL, freq_avg)
        gain_min = damping_value(C1, C2, R1, R2, RL, freq_min)
        gain_max = damping_value(C1, C2, R1, R2, RL, freq_max)

        is_positive_avg = gain_avg > 0.
        is_positive_min = gain_min > 0.
        is_positive_max = gain_max > 0.

        if is_positive_min != is_positive_avg:
            bisection_freq = (freq_min + freq_avg) / 2.
            freq_max = freq_avg
        elif is_positive_max != is_positive_avg:
            bisection_freq = (freq_max + freq_avg) / 2.
            freq_min = freq_avg
        else:
            raise Exception("Bisection was not able to find. Use larger range")

        bisection_gain = damping_value(C1, C2, R1, R2, RL, bisection_freq)

    return bisection_freq


def find_bode_bisection(C1, C2, R1, R2, RL, seek_damping_value):
    freq_min = 0
    freq_max = 10e7

    epsilon = 0.0001
    bisection_freq = float('inf')
    bisection_gain = float('inf')

    while bisection_gain > epsilon:
        freq_avg = (freq_min + freq_max) / 2.

        gain_max = seek_damping_value - bode_characteristic_value(C1, C2, R1, R2, RL, freq_max)
        gain_avg = seek_damping_value - bode_characteristic_value(C1, C2, R1, R2, RL, freq_avg)
        gain_min = seek_damping_value - bode_characteristic_value(C1, C2, R1, R2, RL, freq_min)

        is_positive_avg = gain_avg > 0.
        is_positive_min = gain_min > 0.
        is_positive_max = gain_max > 0.

        if is_positive_min != is_positive_avg:
            bisection_freq = (freq_min + freq_avg) / 2.
            freq_max = freq_avg
        elif is_positive_max != is_positive_avg:
            bisection_freq = (freq_max + freq_avg) / 2.
            freq_min = freq_avg
        else:
            raise Exception("Bisection was not able to find. Use larger range")

        bisection_gain = seek_damping_value - bode_characteristic_value(C1, C2, R1, R2, RL, bisection_freq)

    return bisection_freq


def find_fc_tanget(C1, C2, R1, R2, RL):
    freq_x0 = 3000

    epsilon = 0.00001
    tangent_freq_x0 = freq_x0
    while True:
        damping_freq_x0 = damping_value(C1, C2, R1, R2, RL, tangent_freq_x0)

        freq_x_h = tangent_freq_x0 + epsilon
        damping_freq_x_h = damping_value(C1, C2, R1, R2, RL, freq_x_h)

        derivative_f_x = (damping_freq_x_h - damping_freq_x0) / epsilon

        tanget_freq_x1 = tangent_freq_x0 - damping_freq_x0 / derivative_f_x

        tangent_freq_x0 = tanget_freq_x1
        if damping_value(C1, C2, R1, R2, RL, tanget_freq_x1) < epsilon:
            break

    print(tangent_freq_x0)
    return tangent_freq_x0


def find_fc_newton(C1, C2, R1, R2, RL):
    freq_x0 = 3000

    epsilon = 0.00001
    tangent_freq_x0 = freq_x0
    while True:
        damping_freq_x0 = damping_value(C1, C2, R1, R2, RL, tangent_freq_x0)

        freq_x_h = tangent_freq_x0 - epsilon
        damping_freq_x_h = damping_value(C1, C2, R1, R2, RL, freq_x_h)

        derivative_newton = (damping_freq_x0 - damping_freq_x_h) / epsilon

        tanget_freq_x1 = tangent_freq_x0 - damping_freq_x0 / derivative_newton

        tangent_freq_x0 = tanget_freq_x1
        if damping_value(C1, C2, R1, R2, RL, tanget_freq_x1) < epsilon:
            break

    return tangent_freq_x0


def bode_characteristic_value(C1, C2, R1, R2, RL, freq):
    bode_scalar = 20

    img_part = 1j * 2 * np.pi * freq

    equal_up_c = 1. / (R1 * R2 * C1 * C2)
    equal_down_s2 = img_part ** 2
    equal_down_s1 = img_part * (1. / (R1 * C1) + 1. / (R2 * C1) + 1. / (R2 * C2))
    equal_down_c = 1. / (R1 * R2 * C1 * C2)

    equal = equal_up_c / (equal_down_s2 + equal_down_s1 + equal_down_c)

    return bode_scalar * np.log10(np.abs(equal))


def plot_g(R1, R2, C1, C2, Rl, fmin, fmax):
    frequency_vec = np.arange(start=fmin, stop=fmax, step=100)
    ratio_vec = np.zeros_like(frequency_vec)

    for idx in range(len(frequency_vec)):
        ratio_vec[idx] = bode_characteristic_value(R1, R2, C1, C2, Rl, frequency_vec[idx])
    plt.plot(frequency_vec, ratio_vec, label="Bode characteristic 20log_10")

    seek_damping_value = -3
    zero_gain = find_bode_bisection(R1, R2, C1, C2, Rl, seek_damping_value)
    plt.plot(zero_gain, seek_damping_value, '.', label='damping value {:4.2}'.format(zero_gain, 0))

    plt.xlabel('[Hz]')
    plt.ylabel('[dB]')
    plt.xscale('log')
    plt.xlim(fmin, fmax)
    plt.title('Bode plot')
    plt.legend()
    plt.grid()
    plt.show()

    pass


def part_no_3(C1, C2, R1, R2, RL):
    fmin = 1
    fmax = 10e6
    plot_g(R1, R2, C1, C2, RL, fmin, fmax)


def part_no_2(C1, C2, R1, R2, RL):
    plt.figure(62626)
    frequency_vec, frequency_damping_vec = simulate_damping_ration(C1, C2, R1, R2, RL)
    plt.plot(frequency_vec, frequency_damping_vec, label='F(f)')
    zero_gain = find_fc_bisection(C1, C2, R1, R2, RL)
    plt.plot(zero_gain, 0., '.', label='bisection ({:4.10},{})'.format(zero_gain, 0))
    zero_gain = find_fc_tanget(C1, C2, R1, R2, RL)
    plt.plot(zero_gain, 0., '.', label='tangent ({:4.10},{})'.format(zero_gain, 0))
    zero_gain = find_fc_newton(C1, C2, R1, R2, RL)
    plt.plot(zero_gain, 0., '.', label='newton ({:4.10},{})'.format(zero_gain, 0))
    plt.xlabel('[Hz]')
    plt.ylabel('[dB]')
    plt.title('Gain threshold')
    plt.legend()
    plt.grid()
    plt.show()


def part_no_1(C1, C2, R1, R2, RL):
    for input_vec, frequency, time_vec in generate_input_frequency_vecs(h=0.001, t=0.001):
        t_out, y_out_vec = simulate_rlc_response(C1, C2, R1, R2, RL, input_vec, time_vec)
        plot_results('Simulated output', input_vec, frequency, t_out, [None, y_out_vec])

        t_out, du1_out_vec, du2_out_vec = simulate_rlc_du1_response_euler(
            C1, C2, R1, R2, RL, input_vec, time_vec, frequency)
        plot_results('euler', input_vec, frequency, t_out, [du1_out_vec, du2_out_vec])

        t_out, du1_out_vec, du2_out_vec = simulate_rlc_du1_response_extended_euler(
            C1, C2, R1, R2, RL, input_vec, time_vec, frequency)
        plot_results('extended euler', input_vec, frequency, t_out, [du1_out_vec, du2_out_vec])
        # uncomment use for debug
        # break


def simulate_heat_rect(RL: float, du2_vec, h):
    heat_sum = np.sum(du2_vec)
    heat = heat_sum * (h / RL)
    return heat


def simulate_heat_parabol(RL: float, du2_vec, h):
    heat_rec_sum = simulate_heat_rect(RL, du2_vec, h)

    heat_rec_triangle_no_scalar = 0
    for idx in range(len(du2_vec) - 1):
        heat_rec_triangle_no_scalar += np.abs(du2_vec[idx] - du2_vec[idx + 1])
    heat_rec_triangle = heat_rec_triangle_no_scalar * h / (2 * RL)

    return heat_rec_sum + heat_rec_triangle


def part_no_4(C1, C2, R1, R2, RL):
    h_steps = {
        'h_rect_short': 0.000001,
        'h_rect_long': 0.001,
        'h_parabola_short': 0.000001,
        'h_parabola_long': 0.001
    }

    print('Rect')
    for h in [h_steps['h_rect_short'], h_steps['h_rect_long']]:
        for input_vec, frequency, time_vec in generate_input_frequency_vecs(h=h, t=0.060):
            t_out, _, y_out_vec = simulate_rlc_du1_response_extended_euler(
                C1, C2, R1, R2, RL, input_vec, time_vec, frequency)

            # t_out, y_out_vec = simulate_rlc_response(C1, C2, R1, R2, RL, input_vec, time_vec)
            RL_heat = 1e3
            circut_heat = simulate_heat_rect(RL_heat, y_out_vec, h)
            print('f={} h={} heat={}'.format(frequency, h, circut_heat))
            plt.plot(0, 0, label='Circut heat P={}'.format(circut_heat))


    print('Parabola')
    for h in [h_steps['h_parabola_short'], h_steps['h_parabola_long']]:
        for input_vec, frequency, time_vec in generate_input_frequency_vecs(h=h, t=0.060):
            t_out, _, y_out_vec = simulate_rlc_du1_response_extended_euler(
                C1, C2, R1, R2, RL, input_vec, time_vec, frequency)

            # t_out, y_out_vec = simulate_rlc_response(C1, C2, R1, R2, RL, input_vec, time_vec)
            RL_heat = 1e3
            circut_heat = simulate_heat_parabol(RL_heat, y_out_vec, h)
            print('PARABOL: f={} h={} heat={}'.format(frequency, h, circut_heat))
            plt.plot(0, 0, label='Circut heat P={}'.format(circut_heat))

    plt.show()

def main():
    C1, C2, R1, R2, RL = generate_rlc_parameters()
    part_no_1(C1, C2, R1, R2, RL)
    part_no_2(C1, C2, R1, R2, RL)
    part_no_3(C1, C2, R1, R2, RL)
    part_no_4(C1, C2, R1, R2, RL)

    plt.close('all')


if __name__ == "__main__":
    main()
