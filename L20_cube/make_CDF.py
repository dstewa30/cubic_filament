import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import ks_2samp, mannwhitneyu

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.titlesize'] = 20

bin_edges, cdf, pdf = np.loadtxt('cdf_array.txt', unpack=True)
t_list, fp_cdf = np.loadtxt('fp_cdf_array.txt', unpack=True)

s_bin_edges, s_cdf, s_pdf = np.loadtxt('../sphere45/cdf_array.txt', unpack=True)
s_t_list, s_fp_cdf = np.loadtxt('../sphere45/fp_cdf_array.txt', unpack=True)

plt.figure(tight_layout=True)

plt.plot(bin_edges, cdf, label='Cubic Simulation')
# plt.plot(t_list, fp_cdf, label='Cubic FP')
plt.plot(s_bin_edges, s_cdf, label='Spherical Simulation')
plt.plot(s_t_list, s_fp_cdf, label='First Passage Theory')

diff = cdf - s_cdf
greatest_disp = np.max(np.abs(diff))
print(f"Max difference beteween CDFs: {greatest_disp}")

plt.ylabel("Probability")
plt.xlabel(r"$D\,t$")
plt.title('Cumulative Distribution Function')
plt.legend()
plt.savefig('all_cdf.pdf')

plt.cla()
plt.clf()

plt.plot(bin_edges, pdf, label='Cubic')
plt.plot(s_bin_edges, s_pdf, label='Spherical')
plt.xlabel(r"$D\,t$")
plt.ylabel('Probability')
plt.legend()
plt.title("Simulation Distributions")
plt.savefig('all_pdf.pdf')

statistic, pvalue = ks_2samp(pdf, s_pdf)
print(f"KS Statistic: {statistic}")
print(f"P-value: {pvalue}")

statistic2, pvalue2 = mannwhitneyu(pdf, s_pdf, alternative='two-sided')
print(f"U Statistic: {statistic2}")
print(f"P-value: {pvalue2}")
