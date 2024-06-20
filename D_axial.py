import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def line(x, a, b):
    return a*x+ b 

num_simulations = 1

dimension = 1

for num_sim in range(1, 1+num_simulations):
    t, fv1x, fv1y, fv1z, fv2x, fv2y, fv2z =np.loadtxt('face_vec_pos/face_vec_pos.{}.txt'.format(num_sim), unpack=True)
    t, e1x, e1y, e1z, e2x, e2y, e2z =np.loadtxt('e2e_pos/e2e_pos.{}.txt'.format(num_sim), unpack=True)
    t, axor1x, axor1y, axor1z, axor5x, axor5y, axor5z = np.loadtxt('axial_ortho_pos/axial_ortho_pos.{}.txt'.format(num_sim), unpack=True)
    p4x, p4y, p4z = np.loadtxt("p4_pos/p4_pos.{}.txt".format(num_sim), unpack=True)

    t_max = len(t)
    sample_window_fraction = 0.01
    sample_window = int(t_max * sample_window_fraction)

    # ang_sq_avg_sampled = np.zeros(sample_window)
    OG_Drot_avg_sampled = np.zeros(sample_window)
    pure_ax_avg_sampled = np.zeros(sample_window)
    t0_iter_list = np.arange(0, t_max-sample_window)

    t_shortened = t[:sample_window]
    print("The number of simulations is: ", len(t0_iter_list))

    for t0_i in t0_iter_list:
        # ang_rot_list = np.zeros(sample_window)
        OG_Drot_list = np.zeros(sample_window)
        pure_ax_list = np.zeros(sample_window)
        # fv0 = np.array([fv2x[t0_i] - fv1x[t0_i], fv2y[t0_i] - fv1y[t0_i], fv2z[t0_i] - fv1z[t0_i]])
        e2e0 = np.array([e2x[t0_i] - e1x[t0_i], e2y[t0_i] - e1y[t0_i], e2z[t0_i] - e1z[t0_i]])
        # axor0 = np.array([axor1x[t0_i]-axor5x[t0_i], axor1y[t0_i]-axor5y[t0_i], axor1z[t0_i]-axor5z[t0_i]])
        # axor0 = axor0 / np.linalg.norm(axor0)
        # intermediate_vec0 = np.cross(axor0,fv0)
        # fin_vec0 = np.cross(intermediate_vec0,axor0)

        h0 = np.array([axor1x[t0_i]-axor5x[t0_i], axor1y[t0_i]-axor5y[t0_i], axor1z[t0_i]-axor5z[t0_i]])
        h0 = h0 / np.linalg.norm(h0)
        f0 = np.array([fv2x[t0_i] - fv1x[t0_i], fv2y[t0_i] - fv1y[t0_i], fv2z[t0_i] - fv1z[t0_i]])
        f0 = f0 / np.linalg.norm(f0)
        g0 = np.array([p4x[t0_i] - fv1x[t0_i], p4y[t0_i] - fv1y[t0_i], p4z[t0_i] - fv1z[t0_i]])
        g0 = g0 / np.linalg.norm(g0)

        # standard_to_basis0 = np.linalg.inv(np.column_stack((h0,f0,g0)))

        for i in range(1+t0_i,sample_window+t0_i):

            h_prime = np.array([axor1x[i]-axor5x[i], axor1y[i]-axor5y[i], axor1z[i]-axor5z[i]])
            h_prime = h_prime / np.linalg.norm(h_prime)
            f_prime = np.array([fv2x[i] - fv1x[i], fv2y[i] - fv1y[i], fv2z[i] - fv1z[i]])
            f_prime = f_prime / np.linalg.norm(f_prime)
            g_prime = np.array([p4x[i] - fv1x[i], p4y[i] - fv1y[i], p4z[i] - fv1z[i]])
            g_prime = g_prime / np.linalg.norm(g_prime)

            # prime2standard = np.column_stack((h_prime, f_prime, g_prime))

            # f_prime_basis0 = np.dot(standard_to_basis0, f_prime)
            f_planar_f = np.dot(f_prime, f0)
            f_planar_g = np.dot(f_prime, g0)
            f_planar = (f_planar_f * f0) + (f_planar_g *g0)
            f_planar = f_planar / np.linalg.norm(f_planar)

            phi = np.arccos(np.dot(f_planar,f0)) 
            
            e2e = np.array([e2x[i] - e1x[i], e2y[i] - e1y[i], e2z[i] - e1z[i]])
            e2e_angle = np.arccos(np.dot(e2e0,e2e) / (np.linalg.norm(e2e0) * np.linalg.norm(e2e)))

            # fv = np.array([fv2x[i] - fv1x[i], fv2y[i] - fv1y[i], fv2z[i] - fv1z[i]])
            # axor = np.array([axor1x[i]-axor5x[i], axor1y[i]-axor5y[i], axor1z[i]-axor5z[i]])
            # fv_angle = np.arccos(np.dot(fv0,fv) / (np.linalg.norm(fv0) * np.linalg.norm(fv)))
            # intermediate_vec = np.cross(axor0,fv0)
            # fin_vec0 = np.cross(intermediate_vec0,axor0)
            
            # ang_rot_list[i-t0_i] = fv_angle
            OG_Drot_list[i-t0_i] = e2e_angle
            pure_ax_list[i-t0_i] = phi

        # ang_sq_avg_sampled += (ang_rot_list)**2
        OG_Drot_avg_sampled += (OG_Drot_list)**2
        pure_ax_avg_sampled += (pure_ax_list)**2
    
    # ang_sq_avg_sampled /= len(t0_iter_list)
    OG_Drot_avg_sampled /= len(t0_iter_list)
    pure_ax_avg_sampled /= len(t0_iter_list)

    # with open('data/D_rot_sampled.txt', 'w') as file:
    #     file.write("# t0 \t <phi^2>\n")
    #     for i in range(sample_window):
    #         file.write("{} \t {}\n".format(t_shortened[i], ang_sq_avg_sampled[i]))

    # fit_params, fit_cov = curve_fit(line, t_shortened, ang_sq_avg_sampled)
    fit_params2, fit_cov2 = curve_fit(line, t_shortened, OG_Drot_avg_sampled)
    fit_params3, fit_cov3 = curve_fit(line, t_shortened, pure_ax_avg_sampled)

    # D = fit_params[0]/(2*dimension)
    e2e_D = fit_params2[0]/(2*dimension)
    ax_D = fit_params3[0]/(2*dimension)

    # print('Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params[0], fit_params[1]))
    # print('Diffusion coefficient: D = {:.4f}'.format(D))
    print('e2e Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params2[0], fit_params2[1]))
    print('e2e Diffusion coefficient: e2e_D = {:.4f}'.format(e2e_D))
    print('Pure Axial Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params3[0], fit_params3[1]))
    print('Pure Axial Diffusion coefficient: ax_D = {:.4f}'.format(ax_D))

    # fitline = line(t_shortened, *fit_params)
    e2e_fitline = line(t_shortened, *fit_params2)
    pur_ax_fitline = line(t_shortened, *fit_params3)

#     for i in range(1,len(t)):
#         fv = np.array([fv2x[i] - fv1x[i], fv2y[i] - fv1y[i], fv2z[i] - fv1z[i]])
#         angle = np.arccos(np.dot(fv0,fv) / (np.linalg.norm(fv0) * np.linalg.norm(fv)))
#         axl_ang_rot_list[i] = angle

#     with open('data/axl_ang_rot.{}.txt'.format(num_sim), 'w') as f:
#         f.write('#t\taxl_ang_rot(rad)\n')
#         for i in range(len(t)):
#             f.write('{} \t {}\n'.format(t[i], axl_ang_rot_list[i]))

# t, dphi = np.loadtxt('data/axl_ang_rot.1.txt', unpack=True)

# t_max = len(t)
# sample_window_fraction = 0.02
# sample_window = int(t_max * sample_window_fraction)

# dphi_sq_avg_sampled = np.zeros(sample_window)
# t0_iter_list = np.arange(0, t_max-sample_window)

# t_shortened = t[:sample_window]

# for t0_i in t0_iter_list:
#     dphi_sq_avg_sampled += (dphi[t0_i:t0_i+sample_window])**2

# dphi_sq_avg_sampled /= len(t0_iter_list)
# print(dphi_sq_avg_sampled)

# with open('data/D_rot_sampled.txt', 'w') as file:
#     file.write("# t0 \t <theta^2>\n")
#     for i in range(sample_window):
#         file.write("{} \t {}\n".format(t_shortened[i], dphi_sq_avg_sampled[i]))

# fit_params, fit_cov = curve_fit(line, t_shortened, dphi_sq_avg_sampled)

# D = fit_params[0]/(2*dimension)

# print('Fitted parameters: m = {:.4e}, c = {:.4e}'.format(fit_params[0], fit_params[1]))
# print('Diffusion coefficient: D = {:.4f}'.format(D))

# fitline = line(t_shortened, *fit_params)

# t = np.loadtxt('data/axl_ang_rot.1.txt', usecols=(0,))

# dphi2_cumu = np.zeros(len(t))

# for num_sim in range(1, 1 + num_simulations):
#     t, dphi = np.loadtxt('data/axl_ang_rot.{}.txt'.format(num_sim), unpack=True)

#     dphi2 = dphi**2

#     dphi2_cumu += dphi2

# dphi2_cumu /= num_sim

# p_opt, p_cov = curve_fit(line, t, dphi2_cumu)

# D = p_opt[0] / (2*dimension)

# print("D = {:.4f}".format(D))

# fit = line(t, *p_opt)

# plt.figure(tight_layout=True)
# plt.plot(t_shortened, ang_sq_avg_sampled, label = "MSD", linewidth=1.0, color='black')
# plt.plot(t_shortened, fitline, 'r--', label = "Linear fit", linewidth=1.0)
# plt.xlabel(r'$t/\tau$', fontsize=18)
# plt.ylabel(r'$\langle \Delta \phi^2 \rangle$', fontsize=18)
# plt.xlim(left = 0.0)
# plt.ylim(bottom = 0.0)
# plt.title(r'$D = {:.4f}$'.format(D), fontsize=18)
# plt.legend(fontsize=19)
# plt.savefig('tot_rot_MSD.pdf')

plt.clf
plt.cla
plt.figure(tight_layout=True)
plt.plot(t_shortened, OG_Drot_avg_sampled, label = "MSD", linewidth=1.0, color='black')
plt.plot(t_shortened, e2e_fitline, 'r--', label = "Linear fit", linewidth=1.0)
plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'$\langle \Delta \phi^2 \rangle$', fontsize=18)
plt.xlim(left = 0.0)
plt.ylim(bottom = 0.0)
plt.title(r'$D = {:.4f}$'.format(e2e_D), fontsize=18)
plt.legend(fontsize=19)
plt.savefig('e2e_Drot_MSD.pdf')

plt.clf
plt.cla
plt.figure(tight_layout=True)
plt.plot(t_shortened, pure_ax_avg_sampled, label = "MSD", linewidth=1.0, color='black')
plt.plot(t_shortened, pur_ax_fitline, 'r--', label = "Linear fit", linewidth=1.0)
plt.xlabel(r'$t/\tau$', fontsize=18)
plt.ylabel(r'$\langle \Delta \phi^2 \rangle$', fontsize=18)
plt.xlim(left = 0.0)
plt.ylim(bottom = 0.0)
plt.title(r'$D = {:.4f}$'.format(ax_D), fontsize=18)
plt.legend(fontsize=19)
plt.savefig('axial_Drot_MSD.pdf')