### SAMPLED COM DISPLACEMENT DIFFUSION CONSTANT MSD ###

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

num_simulations = 1

def f(x, m, c):
    return m * x + c

dimension = 3 

for n in range(num_simulations):
    # print('com_pos/com_pos.{}.txt'.format(n+1))
    t, x, y, z = np.loadtxt(
        'com_pos/com_pos.{}.txt'.format(n+1), unpack=True)

    x_displacement = np.zeros_like(x)
    y_displacement = np.zeros_like(y)
    z_displacement = np.zeros_like(z)

    x_displacement = x - x[0]
    y_displacement = y - y[0]
    z_displacement = z - z[0]

    dx =x_displacement
    dy = y_displacement
    dz = z_displacement

    # with open('data/D_com_displacement.{}.txt'.format(n+1), 'w') as file:
    #     file.write("#t\tdx\tdy\tdz\n")
    #     for i in range(len(x_displacement)):
    #         file.write("{}\t{}\t{}\t{}\n".format(t[i],x_displacement[i],y_displacement[i],z_displacement[i]))


    # t, dx, dy, dz = np.loadtxt(
    #     'data/D_com_displacement.{}.txt'.format(n+1), unpack=True)


    t_max = len(t)
    sample_window_fraction = 0.02
    sample_window = int(t_max * sample_window_fraction)
    # print("Total iterations: {}".format(t_max))
    # print("Sample window: {} iterations".format(sample_window))

    # dx_sq_avg_sampled = np.zeros(sample_window)
    # dy_sq_avg_sampled = np.zeros(sample_window)
    ds_sq_avg_sampled = np.zeros(sample_window)

    t0_iter_list = np.arange(0, t_max-sample_window)
    # print("Samples taken: {}".format(len(t0_iter_list)))

    t_shortened = t[:sample_window]

    # print("t0_iter_list: {}".format(t0_iter_list))

    for t0_i in t0_iter_list:
        init_dx = dx[t0_i]
        init_dy = dy[t0_i]
        init_dz = dz[t0_i]
        ds_sq_avg_sampled += (dx[t0_i:t0_i+sample_window]-init_dx)**2 + (dy[t0_i:t0_i+sample_window]-init_dy)**2 + (dz[t0_i:t0_i+sample_window]-init_dz)**2

    ds_sq_avg_sampled /= len(t0_iter_list)

    # with open('data/D_com_sampled.txt', 'w') as file:
    #     file.write("# t0 \t <ds^2>\n")
    #     for i in range(sample_window):
    #         file.write("{} \t {}\n".format(t_shortened[i], ds_sq_avg_sampled[i]))

    fit_params, fit_cov = curve_fit(f, t_shortened, ds_sq_avg_sampled)
    D_com = fit_params[0]/(2*dimension)
    with open('data/D_trans_list.txt', 'w') as file:
        file.write('{}\n'.format(D_com))
    # print('Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params[0], fit_params[1]))
    # print('Diffusion coefficient: D = {:.4e}'.format(fit_params[0]/(2*dimension)))
    # print('or, approximately D = {:.4f}'.format(fit_params[0]/(2*dimension)))

    # fitline = f(t_shortened, *fit_params)

    # plt.figure(tight_layout=True)
    # plt.plot(t_shortened, ds_sq_avg_sampled, 'k-', label='Data')
    # plt.plot(t_shortened, fitline, 'r--', label='Fit')
    # plt.xlabel(r'$t/\tau$', fontsize=18)
    # plt.ylabel(r'$\langle ds^2 \rangle$', fontsize=18)
    # plt.title(r"$D_{{com}} = {:.4f}$".format(fit_params[0]/(2*dimension)))
    # plt.legend(fontsize=14)
    # plt.savefig('plots/D_com_sampled.{}.pdf'.format(n+1))

    # with open('data/D_com_displacementvalue.{}.txt', 'w') as file:
    #     file.write('# D_rot (directly measured)\n')
    #     file.write('{}\n'.format(fit_params[0]/(2*dimension)))