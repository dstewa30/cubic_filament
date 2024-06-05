import numpy as np
from modules.layer import layer
import numpy.linalg as la
from numpy.linalg import norm
from numpy import sqrt, pi, cos, sin


class filament:
    def __init__(self, num_monomers, monomer_diameter, start_pos, heading,num_linker_chain,linker_gap, link_diff):
        self.__num_monomers = num_monomers
        self.__monomer_diameter = monomer_diameter
        self.__start_pos = np.array(start_pos)
        self.__heading = np.array(heading)
        self.__num_linker_chain = num_linker_chain
        self.__linker_gap = linker_gap
        self.__link_diff = link_diff

        self.__layers = []
        self.linker_positions = []
        self.__angles = []
        self.__bonds = []

        self.__generate_filament()

        self.__beads = []
        self.__num_beads = 0

        self.__generate_beads()

    def __generate_filament(self):
        l = layer(self.__monomer_diameter, self.__start_pos, self.__heading)
        self.__layers.append(l)
        a = self.__monomer_diameter
        h, f, g = l.get_basis()
        gap_count = 0

        for k in range(1, self.__num_monomers + 2):
            
            ### To get the positions of the linkers ###
            p = []
            if gap_count % (self.__linker_gap+1) == 0 and gap_count != self.__num_monomers:
                p.append(l.positions[0] + (a*f)/2 + (a*h)/2 - (self.__link_diff*g))
                for bead in range(self.__num_linker_chain-1):
                    placeholder = p[-1]
                    p.append(placeholder-(self.__link_diff*g))

                for linker_pos in p:
                    self.linker_positions.append(linker_pos)

            gap_count += 1

            ### To get bond pairs in the plane of each layer ###
            for j in range((k*4)-3,(k*4)+1):
                if j % 4 != 0:
                    self.__bonds.append([j,j+1])
                else:
                    self.__bonds.append([j,j-3])

            if k != self.__num_monomers + 1:
                ### To get bond pairs between layers ###
                for li in range((k*4)-3,(k*4)+1):
                    self.__bonds.append([li, li+4])
                            
                l = l.make_next_layer()
                self.__layers.append(l)

        ### To get the angles within the plane of each layer ###
        for l in range(len(self.__layers)):
            for counter in range(1,5):
                triplet1 = [(counter)+(4*l), ((counter)%4)+(4*l)+1, ((counter+1)%4)+(4*l)+1]
                self.__angles.append(triplet1)


        for l in range(len(self.__layers)-1):
            for counter in range(1,5):
                ### Each point in each layer form 2 angles involving a point in the layer below it ###
                ### Here we get the 2 angles for each point between the layer and the layer below  ###
                triplet2 = [((counter)%4)+1+(4*l), ((counter)+(4*l)), ((counter+3)%4)+5+(4*l)]
                self.__angles.append(triplet2)
                n=triplet2[0]
                triplet3 = [n,triplet2[1],triplet2[2]]
                triplet3[0] = ((triplet3[0] + 1)%(4) + 1 +(4*l))
                self.__angles.append(triplet3)

                ### These next two triplets then consider the 2 angles involving a point in layer ###
                ### formed with another point in the layer above                                  ###  
                triplet4 = [n,triplet2[2],triplet2[1]]
                triplet4[0] += 4
                self.__angles.append(triplet4)

                n2=triplet3[0]
                triplet5 = [n2,triplet2[2],triplet2[1]]
                triplet5[0] += 4
                self.__angles.append(triplet5)

        ### To calculate bond pairs of linkers ###
        this_linker = 1 + (self.__num_monomers+1)*4
        for gap_count in range(self.__num_monomers):
            # if self.__linker_gap == 1 and gap_count % 2 != 0:
            #     continue
            if gap_count % (self.__linker_gap+1) == 0:
                linker_count = 1 + 4*(gap_count)
                
                for i in range (linker_count, linker_count+2):
                    bondpair = [i, this_linker]
                    bondpair2 = [i+4, this_linker]
                    self.__bonds.append(bondpair)
                    self.__bonds.append(bondpair2)
                for j in range(self.__num_linker_chain-1):
                    bondpair3 = [this_linker,this_linker+1]
                    self.__bonds.append(bondpair3)
                    this_linker += 1
                    # if j == self.__num_linker_chain - 2:
                    #     this_linker += 1
                this_linker += 1
            

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
