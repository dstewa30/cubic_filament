import matplotlib.pyplot as plt
import numpy as np

D_com = np.loadtxt(
    'data/D_trans_list.txt', unpack=True)

D_mean = np.mean(D_com)
D_std_dev = np.std(D_com)
print(D_mean)
print(D_std_dev)

bins_list = np.arange(min(D_com), max(D_com)+1, 1.0)

plt.figure(tight_layout=True)
plt.hist(D_com, bins=bins_list, color='b',edgecolor='k',density=True,rwidth=0.8, align='left')
plt.title(r'Trials = 5000    $\langle D \rangle = {:.2f}$    $\sigma = {:.2f}$'.format(D_mean,D_std_dev))
plt.xlabel(r'$D$', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.savefig('D_com_histogram.pdf')
