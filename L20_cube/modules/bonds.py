import numpy as np
from modules.layer import layer
from modules.filament import filament
import numpy.linalg as la
from modules.positions import get_frame

def join_frame(filament_name: filament, atom_type_template, bond_type_matrix):
    position_list = get_frame(filament_name)
    atom_indx = np.arange(len(position_list))
    print("Atom index: ", atom_indx)
    atom_types = np.zeros_like(atom_indx)
    for ai in range(len(atom_indx)):
        if ai % 4 == 0:
            atom_types[ai] = atom_type_template[0]
        elif ai % 4 == 1:
            atom_types[ai] = atom_type_template[1]
        elif ai % 4 == 2:
            atom_types[ai] = atom_type_template[2]
        elif ai % 4 == 3:
            atom_types[ai] = atom_type_template[3]
    