import numpy as np
from distributions import firstpassage
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.titlesize'] = 20

first_hit_raw, second_hit_raw = np.loadtxt("hitting_times.txt", unpack=True)

first_hit_raw_c = first_hit_raw
second_hit_raw_c = second_hit_raw

max_t = 160.0
D_input = 0.0128

max_t_c = max_t
max_t = max_t*D_input
first_hit_raw = first_hit_raw*D_input
second_hit_raw *= D_input

second_hit = []
first_hit = []

for t in second_hit_raw:
    if (t != max_t):
        second_hit.append(t)

print("Number of attachments found: {} out of {}".format(
    len(second_hit), len(second_hit_raw)))
print("{:3.2f} % successfully attached".format(
    100.0 * len(second_hit) / len(second_hit_raw)))

print("Mean first passage time: {}".format(np.mean(second_hit)))
print("Standard deviation: {}".format(np.std(second_hit)))

mean_time = np.mean(second_hit)

bins_list = np.arange(0, max_t, 5.0*D_input)
# bins_list = "auto"

plt.figure(tight_layout=True)

plt.axvline(x=mean_time, color='r', linestyle='--', label='Mean time: {:.2f}'.format(mean_time))

plt.hist(second_hit, bins=bins_list, histtype='bar', rwidth=0.8,
         color='b', edgecolor='None', density=True, align='right')

plt.xlabel(r"$D\,t$")
plt.ylabel("Frequency")

plt.title('D = {:.4f}'.format(D_input))

plt.xlim(0, max_t)

# Theoretical distribution

theta = 45
theta_rad = np.radians(theta)

t_list = np.linspace(0.1, max_t_c, 1000)
fp_list = np.zeros(len(t_list))

for i in range(len(t_list)):
    fp_list[i] = D_input * firstpassage(D_input * t_list[i], theta_rad)

t_list *= D_input

fp_list_area = np.trapz(fp_list,t_list)

fp_list /= fp_list_area

plt.plot(t_list, fp_list, label=r'First passage $({}\degree)$'.format(
    theta), linestyle="-", color="r")

plt.legend()

plt.savefig("time_hist.pdf")

plt.clf()
plt.cla()

count, bin_edges = np.histogram(second_hit, bins_list)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
fp_pdf = fp_list / sum(fp_list)
fp_cdf = np.cumsum(fp_pdf)

bin_edges = bin_edges[:-1]

plt.plot(bin_edges, cdf, label='Simulation')
plt.plot(t_list, fp_cdf, label='First Pass Theory')


plt.ylabel("Probability")
plt.xlabel(r"$D\,t$")
plt.title('Cumulative Density Function')
plt.legend()
plt.savefig('cdf.pdf')

plt.clf()
plt.cla()


for t in first_hit_raw:
    if (t != max_t):
        first_hit.append(t)

print("Number of first attachments found: {} out of {}".format(
    len(first_hit), len(first_hit_raw)))

print("{:3.2f} % successfully attached".format(
    100.0 * len(first_hit) / len(first_hit_raw)))

mean_one_linker_time = np.mean(first_hit)

plt.figure(tight_layout=True)

plt.axvline(x=mean_one_linker_time, color='r', linestyle='--', label='Mean time: {:.2f}'.format(mean_one_linker_time))

plt.hist(first_hit, bins=bins_list, histtype='bar', rwidth=0.8,
         color='b', edgecolor='None', density=True, align='right')

plt.plot(t_list, fp_list, label=r'First passage $({}\degree)$'.format(
    theta), linestyle="-", color="r")

plt.xlabel(r"$D\,t$")
plt.ylabel("Frequency")

plt.title('D = {:.4f}'.format(D_input))

plt.xlim(0, max_t)

plt.legend()

plt.savefig("time_hist_first.pdf")

plt.clf()
plt.cla()

count, bin_edges = np.histogram(first_hit, bins_list)
pdf = count / sum(count)
cdf = np.cumsum(pdf)

bin_edges = bin_edges[:-1]

plt.plot(bin_edges, cdf, label='Simulation')
plt.plot(t_list, fp_cdf, label='First Pass Theory')


plt.ylabel("Probability")
plt.xlabel(r"$D\,t$")
plt.title('Cumulative Density Function')
plt.legend()
plt.savefig('cdf_first.pdf')

with open('cdf_array.txt', 'w') as file:
    for i in range(len(cdf)):
        file.write('{}\t{}\t{}\n'.format(bin_edges[i], cdf[i], pdf[i]))

with open('fp_cdf_array.txt', 'w') as file:
    for i in range(len(fp_cdf)):
        file.write('{}\t{}\n'.format(t_list[i], fp_cdf[i]))


