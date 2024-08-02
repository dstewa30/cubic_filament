import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x + b

dimension = 1

runmax = 1050

t = np.loadtxt('data/ang_deviation.1.txt', usecols=(0,))

dtheta2_cumu = np.zeros(len(t))

for run_i_m1 in range(0, runmax):

    run_i = run_i_m1 + 1
    
    # print("Processing run {}".format(run_i))

    t, ang_dev_rad, ang_dev_deg = np.loadtxt(
        'data/ang_deviation.{}.txt'.format(run_i), unpack=True)

    theta = ang_dev_rad

    theta0 = theta[0]

    dtheta = theta - theta0

    dtheta2 = dtheta**2

    dtheta2_cumu += dtheta2

dtheta2_cumu /= runmax

p_opt, p_cov = curve_fit(f, t, dtheta2_cumu)

D = p_opt[0] / (2*dimension)

print("D = {:.4f}".format(D))

fit = f(t, *p_opt)

plt.figure(tight_layout=True)
plt.plot(t, dtheta2_cumu, label = "MSD", linewidth=1.0, color='black')
plt.plot(t, fit, label = "Linear fit", linewidth=1.0, color='red', linestyle='--')
plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'$\langle \Delta \theta^2 \rangle$', fontsize=18)
plt.xlim(left = 0.0)
plt.ylim(bottom = 0.0)
plt.title(r'$D = {:.4f}$'.format(D), fontsize=18)
plt.legend(fontsize=19)
plt.savefig('ang_MSD.pdf')

avg_e2e = np.zeros(len(t))
mean_e2e = 0

for n in range(1,runmax+1):
    # Load data
    # print('Loading e2e data from run_{}'.format(n))
    t, e2e = np.loadtxt(
        'e2e_pos/e2e_dist.{}.txt'.format(n), unpack=True)
    avg_e2e += e2e
    
    mean_e2e += np.mean(e2e[10:])

avg_e2e = avg_e2e[10:]
avg_e2e = avg_e2e/runmax
mean_e2e = mean_e2e/runmax

mean_line = mean_e2e*np.ones(len(t[10:]))

plt.figure(tight_layout=True)

plt.plot(t[10:], avg_e2e, 'k-', label='Averaged length', linewidth=1.0)
plt.plot(t[10:], mean_line, 'r--', label='Mean length ({:.1f})'.format(mean_e2e), linewidth=1.0)

plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'Averaged end to end distance', fontsize=18)

plt.legend()

plt.savefig('plots/avg_e2e_dist.pdf')