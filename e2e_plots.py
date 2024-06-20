import numpy as np
import matplotlib.pyplot as plt

info = np.loadtxt('info.txt')
num_simulations = 1050

num_monomers = int(info[0])
# num_linkers = int(info[1])
# num_skip = int(info[2])

for num_sim in range(1, 1+num_simulations):
    t, e1x, e1y, e1z, e2x, e2y, e2z = np.loadtxt('e2e_pos/e2e_pos.{}.txt'.format(num_sim), unpack=True)

    hv0 = np.array([e1x[0] - e2x[0], e1y[0] - e2y[0], e1z[0] - e2z[0]])

    ang_deviation_list = np.zeros(len(t))

    for i in range(len(t)):
        hv = np.array([e1x[i] - e2x[i], e1y[i] - e2y[i], e1z[i] - e2z[i]])
        angle = np.arccos(np.dot(hv0, hv) / ((np.linalg.norm(hv0) * np.linalg.norm(hv))))
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

    # plt.clf()
    # plt.cla()

    # plt.figure(tight_layout=True)

    # plt.plot(t, e2e_dist_list, 'k-', linewidth=0.2, label='End to end distance')
    # # plt.scatter(t, e2e_dist_list, s=5, label='End to end distance', color='k', marker='.', alpha=0.7, edgecolors='none')

    # plt.plot(t, e2e_mean_line, 'r--', linewidth=2, label='Mean distance', alpha=0.7)
    # plt.xlabel(r'$t/\tau$', fontsize=18)
    # plt.ylabel("End to end distance", fontsize=18)
    # plt.xlim(0, max(t))
    # plt.legend(loc='best', markerscale=2)

    # plt.savefig('e2e_dist.{}.pdf'.format(num_sim))

        