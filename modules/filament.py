import numpy as np
from modules.layer import layer
import numpy.linalg as la
from numpy.linalg import norm
from numpy import sqrt, pi, cos, sin


class filament:
    def __init__(self, num_monomers, monomer_diameter, start_pos, heading, num_linkers):
        self.__num_monomers = num_monomers
        self.__monomer_diameter = monomer_diameter
        self.__start_pos = np.array(start_pos)
        self.__heading = np.array(heading)
        self.__num_linkers = num_linkers

        self.__layers = []
        self.linker_positions = []
        self.basis_sets = []
        self.__angles = []
        self.__bonds = []

        self.__generate_filament()

        self.__beads = []
        self.__num_beads = 0

        self.__generate_beads()

    def __generate_filament(self):
        l = layer(self.__monomer_diameter, self.__start_pos, self.__heading)
        self.__layers.append(l)
        a = (self.__monomer_diameter)
        h, f, g = l.get_basis()
        self.basis_sets.append([h,f,g])
        gap_count = 0

        ### To get the positions of the linkers ###
        linker_gap = self.__num_monomers // self.__num_linkers

        link_height = 2 # This is arbitrary due to bonds

        self.linker_positions.append(l.positions[0] + (a*f)/2 + ((self.__num_monomers*a-(a/2))*h) +  (link_height + a)*g)
        placeholder = self.linker_positions[0]
        for i in range(1, self.__num_linkers):
            self.linker_positions.append(placeholder - (a*i*(linker_gap))*h)
            
        for k in range(1, self.__num_monomers + 1):
            
            l = l.make_next_layer()
            self.__layers.append(l)

        ## CURVATURE BONDS !!    
        # 4.2 depth bonds
        for k in range(1, (self.__num_monomers*4)+4, 2):
            self.__bonds.append([1,k,k+1])

        # 2L bonds (diagnonal sides of trapezoid)
        for k in range(2, (self.__num_monomers*4)+4, 4):
            self.__bonds.append([2,k,k+1])
            self.__bonds.append([2,k+2,k-1])

        # Short base trapezoid bonds -- a2
        for k in range(1, (self.__num_monomers*4), 4):
            self.__bonds.append([3,k,k+4])
            self.__bonds.append([3,k+1,k+5])

        # Long base trapezoid bonds -- a1
        for k in range(3, (self.__num_monomers*4), 4):
            self.__bonds.append([4,k,k+4])
            self.__bonds.append([4,k+1,k+5])

        num_beads_f = (self.__num_monomers*4) + 4

        # LINKER BONDS
        # Near Bonds
        for k, i in zip(range(num_beads_f+1, num_beads_f + 1 + self.__num_linkers), range((num_beads_f - 5), 1, -4*linker_gap)):  
            self.__bonds.append([5,k,i])
            self.__bonds.append([5,k,i+4])
            self.__bonds.append([5,k,i+5])
            self.__bonds.append([5,k,i+1])

        # Far Bonds
        for k, i in zip(range(num_beads_f+1, num_beads_f + 1 + self.__num_linkers), range((num_beads_f - 6), 1, -4*linker_gap)):
            bond_type = 6
            self.__bonds.append([bond_type,k,i])
            self.__bonds.append([bond_type,k,i+4])
            self.__bonds.append([bond_type,k,i+3])
            self.__bonds.append([bond_type,k,i-1])


        ## CURVATURE ANGLES
        # Perpendicular Angles
        # Rect Prism Depth face
        for k in range(1, (self.__num_monomers*4)+4, 4):
            self.__angles.append([1,k,k+1,k+2])
            self.__angles.append([1,k+1,k+2,k+3])
            self.__angles.append([1,k+2,k+3,k])
            self.__angles.append([1,k+3,k,k+1])

        # Long base top face    
        for k in range(3, (self.__num_monomers*4), 4):
            self.__angles.append([1,k,k+4,k+5])
            self.__angles.append([1,k+4,k+5,k+1])
            self.__angles.append([1,k+5,k+1,k])
            self.__angles.append([1,k+1,k,k+4])

        # Short base bottom face
        for k in range(2, (self.__num_monomers*4), 4):
            self.__angles.append([1,k,k+4,k+3])
            self.__angles.append([1,k+4,k+3,k-1])
            self.__angles.append([1,k+3,k-1,k])
            self.__angles.append([1,k-1,k,k+4])

        # Trap face front
        for k in range(3, (self.__num_monomers*4), 4):
            self.__angles.append([2,k,k-1,k+3])
            self.__angles.append([2,k-1,k+3,k+4])
            self.__angles.append([3,k+3,k+4,k])
            self.__angles.append([3,k+4,k,k-1])

        # Trap face back
        for k in range(4, (self.__num_monomers*4)+1, 4):
            self.__angles.append([2,k,k-3,k+1])
            self.__angles.append([2,k-3,k+1,k+4])
            self.__angles.append([3,k+1,k+4,k])
            self.__angles.append([3,k+4,k,k-3])

        ### To calculate bond pairs of linkers ###


        ### To calculate bond pairs of linkers ###
        # this_linker = 1 + (self.__num_monomers+1)*4
        # for gap_count in range(self.__num_monomers):
        #     # if self.__linker_gap == 1 and gap_count % 2 != 0:
        #     #     continue
        #     if gap_count % (self.__linker_gap+1) == 0:
        #         linker_count = 1 + 4*(gap_count)
                
        #         for i in range (linker_count, linker_count+2):
        #             bondpair = [i, this_linker]
        #             bondpair2 = [i+4, this_linker]
        #             self.__bonds.append(bondpair)
        #             self.__bonds.append(bondpair2)
        #         for j in range(self.__num_linker_chain-1):
        #             bondpair3 = [this_linker,this_linker+1]
        #             self.__bonds.append(bondpair3)
        #             this_linker += 1
        #             # if j == self.__num_linker_chain - 2:
        #             #     this_linker += 1
        #         this_linker += 1

        # this_linker = 1 + (self.__num_monomers+1)*4
        # for gap_count in range(self.__num_monomers):
        #     if gap_count % (self.__linker_gap+1) == 0:
        #         for i in range(1+(4*(self.__num_monomers-1))-(gap_count*4),9+(4*(self.__num_monomers-1))-(gap_count*4)):
        #             bondpair = [i,this_linker]
        #             self.__bonds.append(bondpair)
        #         this_linker += 1 
                    

            

    def __generate_beads(self):
        for li in range(1, len(self.__layers)):
            plist = np.zeros(3)
            for p in self.__layers[li].positions:
                p = np.array(p)
                plist += p
            for p in self.__layers[li - 1].positions:
                p = np.array(p)
                plist += p
            plist /= 8

            self.__beads.append(plist)
            self.__num_beads += 1

    @property
    def layers(self):
        return self.__layers

    @property
    def beads(self):
        return self.__beads

    @property
    def num_beads(self):
        return self.__num_beads
    
    @property
    def monomer_diameter(self):
        return self.__monomer_diameter
    
    @property
    def bonds(self):
        return self.__bonds
    
    @property
    def angles(self):
        return self.__angles
