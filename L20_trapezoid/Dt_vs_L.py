import numpy as np

D_com = np.loadtxt(
    'data/D_trans_list.txt', unpack=True)

D_com = D_com / D_com[0]
