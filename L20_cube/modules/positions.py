import numpy as np
from modules.layer import layer
from modules.filament import filament
import numpy.linalg as la

def get_frame(filament: filament):
    pos_list = np.zeros((len(filament.layers) * 4, 3))
    for i_layer, layers in enumerate(filament.layers):
        this_layer_positions = layers.positions
        for i_pos, pos in enumerate(this_layer_positions):
            atom_num = i_layer * 4 + i_pos
            pos_list[atom_num] = pos
    return pos_list
    
        
                