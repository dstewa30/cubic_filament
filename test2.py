import numpy as np
from modules.filament import filament
from modules.makexyz import dump_filament
from modules.layer import layer

n=10
n_linker_chain = 1
linker_gap = 3
height = 1
positions = []


start = [0,0,0]
head = [0,0,1]
f1 = filament(n, 1, start, head, n_linker_chain, linker_gap, height)

num_layers = len(f1.layers)
chain = 1
thisatom = 1
monomer_atom = 1
linker_atom = 2

for i in range(num_layers):
    for j in range(len(f1.layers[i].positions)):
        px = f1.layers[i].positions[j][0]
        py = f1.layers[i].positions[j][1]
        pz = f1.layers[i].positions[j][2]
        positions.append([px, py, pz])

for pos in f1.linker_positions:
    px = pos[0]
    py = pos[1]
    pz = pos[2]
    positions.append([px, py, pz])

# print(positions)
print(positions)

# print(f1.angles)
# dump_filament("test.xyz", [f1], True)