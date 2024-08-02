import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x + b

dimension = 3

n_atoms, n_linkers, n_link_skip = np.loadtxt('info.txt')

plt.figure(tight_layout=True)

t, ds2 = np.loadtxt('com_disp_avg.txt', unpack=True)

p_opt, p_cov = curve_fit(f, t, ds2)

D = p_opt[0] / (2*dimension)

print("D = {:.4f}".format(D))

fit = f(t, *p_opt)

plt.plot(t, ds2, label = "MSD", linewidth=1.0, color='black')
plt.plot(t, fit, label = "Linear fit", linewidth=1.0, color='red', linestyle='--')
plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'$\langle \Delta s^2 \rangle$', fontsize=18)
plt.xlim(left = 0.0)
plt.ylim(bottom = 0.0)
plt.title(r'$D = {:.2f}$'.format(D), fontsize=18)
plt.legend(fontsize=19)
plt.savefig('com_MSD.pdf')

with open('D_com_measurment.txt', 'w') as f:
    f.write('#D_com_translation\n')
    f.write('{:.4f}\n'.format(D))
