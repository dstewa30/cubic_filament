import numpy as np
import matplotlib.pyplot as plt

n_atoms, n_linkers, n_link_skip = np.loadtxt('info.txt')

runmax = 1

plt.figure(tight_layout=True)

t = np.loadtxt('com_pos/com_pos.1.txt', usecols=(0,))

ds2_cumu = np.zeros(len(t))

for run_i_m1 in range(0, runmax):

    plt.clf()
    plt.cla()

    run_i = run_i_m1 + 1
    # print("="*50)
    print("Processing run {}".format(run_i))

    t, cmx, cmy, cmz = np.loadtxt(
        'com_pos/com_pos.{}.txt'.format(run_i), unpack=True)

    cmx0, cmy0, cmz0 = cmx[0], cmy[0], cmz[0]

    dx = cmx - cmx0
    dy = cmy - cmy0
    dz = cmz - cmz0

    ds2 = (dx**2 + dy**2 + dz**2)
    ds = np.sqrt(ds2)

    ds2_cumu += ds2

    plt.plot(t, ds, linewidth=1.0, label="CoM displacement", color='black')
    plt.xlabel(r'$t/\tau$', fontsize=18)
    plt.ylabel(r'$\Delta s$', fontsize=18)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.legend(fontsize=19)
    plt.savefig('com_disp.{}.pdf'.format(run_i))

plt.clf()
plt.cla()

ds2_cumu /= runmax

with open('com_disp_avg.txt', 'w') as f:
    for i in range(len(t)):
        f.write('{} {}\n'.format(t[i], ds2_cumu[i]))
