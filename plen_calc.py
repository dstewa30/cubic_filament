import numpy as np
import matplotlib.pyplot as plt

n_monomers = 20

bead_pairs = []

# To write the code into lammps

# for i in range(3,(n_monomers*4)+4, 4):
#     bead_pairs.append(i)

# with open('plength_file.txt', 'w') as file:
#     for bead in range(1, len(bead_pairs)):

#         file.write(f'variable plen{bead}x equal x[{bead_pairs[bead]}]-x[{bead_pairs[bead-1]}]\n')
#         file.write(f'variable plen{bead}y equal y[{bead_pairs[bead]}]-y[{bead_pairs[bead-1]}]\n')
#         file.write(f'variable plen{bead}z equal z[{bead_pairs[bead]}]-z[{bead_pairs[bead-1]}]\n')
#     for bead in range(1,len(bead_pairs)):
#         file.write('fix plen' + str(bead) + ' all print ${recordinterval} "${tsteps} ${plen' + str(bead) + 'x} ${plen' + str(bead) + 'y} ${plen' + str(bead) + 'z}" file plen/plen.' + str(bead) + '.txt screen no\n')




plens_list =[]


for i in range(1,n_monomers+1):
    t, x, y, z = np.loadtxt(f"plen/plen.{i}.txt", unpack=True)

    plen_i = np.array([x,y,z])
    plens_list.append(plen_i)

vec1 = []
for k in range(3):
    vec1.append(plens_list[0][k])
vec1 = np.array(vec1)
dot_list = []

for i in range(1,n_monomers):
    dot_avg = 0
    vec2 = []
    for k in range(3):
        vec2.append(plens_list[i][k])
    
    vec2 = np.array(vec2)
    
    for k in range(len(t)):
        vec_t1 = []
        vec_t2 = []
        for j in range(3):
            vec_t1.append(vec1[j][k])
            vec_t2.append(vec2[j][k])
        vec_t1 = np.array(vec_t1)
        vec_t1 /= np.linalg.norm(vec_t1)
        vec_t2 = np.array(vec_t2)
        vec_t2 /= np.linalg.norm(vec_t2)

        # print("Vec_t1: ", vec_t1)
        # print("Vec_t2:", vec_t2)

        dot_prod = np.dot(vec_t1,vec_t2)
        # print("Dot product: ", dot_prod)
        dot_avg += dot_prod
    
    dot_avg /= len(t)
    dot_list.append(dot_avg)

dist_list = [i for i in range(2,21)]

plt.figure(tight_layout=True)
plt.plot(dist_list, np.log(dot_list))
plt.xlabel('Length')
plt.ylabel('Average Dot Product')
plt.title("Persistence Length")

plt.show()
