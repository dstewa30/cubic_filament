import numpy as np
import matplotlib.pyplot as plt
import warnings
# import seaborn as sns
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['axes.titlesize'] = 20

# np.seterr(all='warn')
# warnings.filterwarnings("error")
info = np.loadtxt('info.txt')
num_simulations = 1
R = 100
phi = 0.84

num_monomers = int(info[0])
# num_linkers = int(info[1])
# num_skip = int(info[2])

for num_sim in range(1, 1+num_simulations):
    t, e1x_o, e1y_o, e1z_o, e1x_i, e1y_i, e1z_i, e2x_o, e2y_o, e2z_o, e2x_i, e2y_i, e2z_i  = np.loadtxt('e2e_pos/e2e_pos.{}.txt'.format(num_sim), unpack=True)
    e1x = (e1x_o + e1x_i) / 2
    e1y = (e1y_o + e1y_i) / 2
    e1z = (e1z_o + e1z_i) / 2

    e2x = (e2x_o + e2x_i) / 2
    e2y = (e2y_o + e2y_i) / 2
    e2z = (e2z_o + e2z_i) / 2

    hv0 = np.array([e1x[0] - e2x[0], e1y[0] - e2y[0], e1z[0] - e2z[0]])

    ang_deviation_list = np.zeros(len(t))
    
    for i in range(len(t)):
        hv = np.array([e1x[i] - e2x[i], e1y[i] - e2y[i], e1z[i] - e2z[i]])
        # print(f"hv0: {hv0} and hv: {hv}")
        # print(i)
        try:
            angle = np.arccos(np.dot(hv0, hv) / ((np.linalg.norm(hv0) * np.linalg.norm(hv))))  
        except:
            # Vectors are the same, completely parallel
            angle = 0
        ang_deviation_list[i] = angle
        

    with open('data/ang_deviation.{}.txt'.format(num_sim), 'w') as f:
        f.write('#t ang_deviation(rad) ang_deviation(deg)\n')
        for i in range(len(t)):
            f.write('{}\t{:.4f}\t{:.4f}\n'.format(t[i], ang_deviation_list[i], np.rad2deg(ang_deviation_list[i])))

    # t, ang_deviation_list_rad, ang_deviation_list_deg = np.loadtxt('data/ang_deviation.txt', unpack=True)

    # plt.figure(tight_layout=True)

    # plt.plot(t, ang_deviation_list_deg, 'k-', linewidth=0.5, label=r'$\Delta\theta$')
    # plt.xlabel(r'$t/\tau$', fontsize=18)
    # plt.ylabel(r'$\Delta\theta$', fontsize=18)

    # plt.savefig('plots/ang_deviation.pdf')

    with open('e2e_pos/e2e_dist.{}.txt'.format(num_sim), 'w') as f:
        f.write('#t e2e_dist\n')
        for i in range(len(t)):
            hv = np.array([e1x[i] - e2x[i], e1y[i] - e2y[i], e1z[i] - e2z[i]])
            f.write('{}\t{:.4f}\n'.format(t[i], np.linalg.norm(hv)))

    t, e2e_dist_list = np.loadtxt('e2e_pos/e2e_dist.{}.txt'.format(num_sim), unpack=True)
    # print("End to end distance: ")
    # print("Mean: {:.4f}".format(np.mean(e2e_dist_list)))
    # print("Std: {:.4f}".format(np.std(e2e_dist_list)))

    

    e2e_mean_line = np.mean(e2e_dist_list) * np.ones(len(t))
    e2e_theoretical = 2*R*np.sin(phi/2) * np.ones(len(t))
    contour_length = 84 *np.ones(len(t))

    # plt.clf()
    # plt.cla()

    plt.figure(tight_layout=True)

    plt.hist(e2e_dist_list, bins='auto', histtype='bar', rwidth=0.8,
         color='b', edgecolor='None', density=True, align='right')
    plt.xlabel('End to End Length (nm)')
    plt.ylabel('Frequency')
    distro_mean = np.mean(e2e_dist_list)
    distro_std = np.std(e2e_dist_list)
    plt.title(r'$\langle L \rangle = {:.2f}, \ \sigma = {:.2f}$'.format(distro_mean, distro_std))
    plt.savefig('e2e_distro.pdf')

    plt.clf()
    plt.cla()

    # count, bin_edges = np.histogram(second_hit, bins_list)
    # pdf = count / sum(count)

    plt.plot(t, e2e_dist_list, 'k-', linewidth=0.2, label='End to end distance')
    # plt.scatter(t, e2e_dist_list, s=5, label='End to end distance', color='k', marker='.', alpha=0.7, edgecolors='none')

    plt.plot(t, e2e_mean_line, 'r--', linewidth=2, label='Mean distance: {:.2f}'.format(e2e_mean_line[0]), alpha=0.7)
    plt.plot(t, e2e_theoretical, 'b--', linewidth=2, label='Theoretical Distance: {:.2f}'.format(e2e_theoretical[0]), alpha=0.7)
    # plt.plot(t, contour_length, 'y--', linewidth=2, label='Contour Length: {}'.format(contour_length[0]), alpha=0.7)
    plt.xlabel(r'$t/\tau$', fontsize=18)
    plt.ylabel("End to end distance (nm)", fontsize=18)
    plt.xlim(0, max(t))
    plt.legend(loc='best', markerscale=2)
    plt.savefig('e2e_dist.{}.pdf'.format(num_sim))

        