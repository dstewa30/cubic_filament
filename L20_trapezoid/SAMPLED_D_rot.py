import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def f(x, m, c):
    return m * x + c
D_list= []
runmax = 1008
t_list =[]
plt.figure(tight_layout=True)
for _ in range(2,runmax+1):
    t, ang_dev_rad, ang_dev_deg = np.loadtxt(
        'data/ang_deviation.{}.txt'.format(_), unpack=True)
    print(_)

    if _ == 2:
        t_max = 4800
        sample_window_fraction = 0.01
        sample_window = int(t_max * sample_window_fraction)
        print("Total iterations: {}".format(t_max))
        print("Sample window: {} iterations".format(sample_window))
        t0_iter_list = np.arange(0, t_max-sample_window)
        print("Samples taken: {}".format(len(t0_iter_list)))

    ang_dev_sq_avg_sampled = np.zeros(sample_window)
    t_shortened = t[:sample_window]

    # print("t0_iter_list: {}".format(t0_iter_list))

    for t0_i in t0_iter_list:
        init_angle = ang_dev_rad[t0_i]
        ang_dev_sq_avg_sampled += (ang_dev_rad[t0_i:t0_i+sample_window]-init_angle)**2

    ang_dev_sq_avg_sampled /= len(t0_iter_list)

    # plt.plot(t_shortened, ang_dev_sq_avg_sampled, 'k-', label='Data')
    # plt.show()

    # with open('data/D_rot_sampled.{}.txt'.format(_), 'w') as file:
    #     file.write("# t0 \t <theta^2>\n")
    #     for i in range(sample_window):
    #         file.write("{} \t {}\n".format(t_shortened[i], ang_dev_sq_avg_sampled[i]))

    fit_params, fit_cov = curve_fit(f, t_shortened, ang_dev_sq_avg_sampled)
    D = fit_params[0]/(2)
    D_list.append(D)
    # print('Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params[0], fit_params[1]))
    # print('Diffusion coefficient: D = {:.4e}'.format(fit_params[0]/(2)))
    # print('or, approximately D = {:.4f}'.format(fit_params[0]/(2)))

    fitline = f(t_shortened, *fit_params)
    
    if _ == 2:
        plt.clf()
        plt.cla()
        plt.plot(t_shortened, ang_dev_sq_avg_sampled, 'k-', label='MSD')
        plt.plot(t_shortened, fitline, 'r--', label='Fit')
        plt.xlabel(r'$t/\tau$', fontsize=18)
        plt.ylabel(r'$\langle \theta^2 \rangle$', fontsize=18)
        plt.title(r"$D_{{rot}} = {:.4f}$".format(fit_params[0]/(2)))
        plt.legend(fontsize=14)
        plt.savefig('plots/D_rot_sampled.{}.pdf'.format(_))

    # with open('data/D_rot_value.txt', 'w') as file:
    #     file.write('# D_rot (directly measured)\n')
    #     file.write('{}\n'.format(fit_params[0]/(2)))

D_avg = np.average(D_list)
D_std = np.std(D_list)
print(D_avg)

plt.clf()
plt.cla()
plt.hist(D_list, bins='auto', color='b', alpha=1.0, edgecolor='k', density=True, rwidth=0.8, align = 'left')
plt.xlabel(r'$D$', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.title(r'Trials = {1}    $\langle D \rangle = {0:.4f}$    $\sigma = {2:.4f}$'.format(D_avg, 1008, D_std))


plt.savefig('plots/sampled_D_hist.pdf')
