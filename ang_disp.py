import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x + b

dimension = 1

runmax = 1

t = np.loadtxt('data/ang_deviation.1.txt', usecols=(0,))

dtheta2_cumu = np.zeros(len(t))

for run_i_m1 in range(0, runmax):

    run_i = run_i_m1 + 1
    
    print("Processing run {}".format(run_i))

    t, ang_dev_rad, ang_dev_deg = np.loadtxt(
        'data/ang_deviation.{}.txt'.format(run_i), unpack=True)

    theta = ang_dev_rad

    theta0 = theta[0]

    dtheta = theta - theta0

    dtheta2 = dtheta**2

    dtheta2_cumu += dtheta2

dtheta2_cumu /= runmax

p_opt, p_cov = curve_fit(f, t, dtheta2)

D = p_opt[0] / (2*dimension)

print("D = {:.4f}".format(D))

fit = f(t, *p_opt)

plt.figure(tight_layout=True)
plt.plot(t, dtheta2, label = "MSD", linewidth=1.0, color='black')
plt.plot(t, fit, label = "Linear fit", linewidth=1.0, color='red', linestyle='--')
plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'$\langle \Delta \theta^2 \rangle$', fontsize=18)
plt.xlim(left = 0.0)
plt.ylim(bottom = 0.0)
plt.title(r'$D = {:.4f}$'.format(D), fontsize=18)
plt.legend(fontsize=19)
plt.savefig('ang_MSD.pdf')