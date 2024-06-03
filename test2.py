import numpy as np
from modules.filament import filament
from modules.makexyz import dump_filament
from modules.layer import layer

n=2

start = [0,0,0]
head = [0,0,1]
f1 = filament(n, 1, start, head,2)

for i in range(len(f1.layers)):
    for j in range(len(f1.layers[0].positions)):
        print(f1.layers[i].positions[j])

# print(f1.angles)
# dump_filament("test.xyz", [f1], True)