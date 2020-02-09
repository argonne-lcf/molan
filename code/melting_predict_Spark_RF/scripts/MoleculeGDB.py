###!i/usr/bin/env python
#
#import os
#import sys
#import glob
#import getopt
#import logging
#import math
#import aims_inp
#
#import pickle
#
#import random

#import nwchem_inp

#from pylab import *
#from numpy import *

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt


#import numpy as np
#from numpy.linalg import inv

#from math import sin, cos , acos , pi


#cclibfrom cclib.parser.utils import PeriodicTable
#cclibfrom cclib.parser import ccopen

#from munch import Munch


toang=1.889725989
tobohr=27.21138505

delta_disp = 0.01 #dft


proplist=['tag','index','A','B','C','mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']

class MoleculeGDB9():

    def __init__(self):
        self.nats = []
        self.energy  = []
        self.label  = []
        self.coord  = []    # creates a new empty list for each dog
        self.force = []    # creates a new empty list for each dog
        self.forcewbsse = []    # creates a new empty list for each dog
        self.forcemons = []    # creates a new empty list for each dog
        self.energymon1  = []
        self.energymon2  = []
        self.bsse  = []
        self.energywbsse  = []
        self.coords = {}
        self.json = {}

#Line       Content
#----       -------
#1          Number of atoms na
#2          Properties 1-17 (see below)
#3,...,na+2 Element type, coordinate (x,y,z) (Angstrom), and Mulliken partial charge (e) of atom
#na+3       Frequencies (3na-5 or 3na-6)
#na+4       SMILES from GDB9 and for relaxed geometry
#na+5       InChI for GDB9 and for relaxed geometry
#
#The properties stored in the second line of each file:
#
#I.  Property  Unit         Description
#--  --------  -----------  --------------
# 1  tag       -            "gdb9"; string constant to ease extraction via grep
# 2  index     -            Consecutive, 1-based integer identifier of molecule
# 3  A         GHz          Rotational constant A
# 4  B         GHz          Rotational constant B
# 5  C         GHz          Rotational constant C
# 6  mu        Debye        Dipole moment
# 7  alpha     Bohr^3       Isotropic polarizability
# 8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
# 9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
#10  gap       Hartree      Gap, difference between LUMO and HOMO
#11  r2        Bohr^2       Electronic spatial extent
#12  zpve      Hartree      Zero point vibrational energy
#13  U0        Hartree      Internal energy at 0 K
#14  U         Hartree      Internal energy at 298.15 K
#15  H         Hartree      Enthalpy at 298.15 K
#16  G         Hartree      Free energy at 298.15 K
#17  Cv        cal/(mol K)  Heat capacity at 298.15 K


    def readxyz(self, filename):
        f = open(filename)
        j = 0
        while 1:
            atomcoords = []
            forces = []
            labels = []
            coordat = []
            e = 0

            line = f.readline()
            if not line: break
            broken = line.split()
            nat = int(broken[0])
            line = f.readline() #17 properties
            infoline = line.split()
            self.prop=dict(zip(proplist,infoline))
            #e = float(infoline[0])
            e=0.0
            i = 0
            for i in range(nat):
                line = f.readline()
                if not line: break
                xcoord = line.split()
                labels.append(xcoord[0])
                atomcoords.append(map(float, xcoord[1:4]))
                atom={}
                atom=dict(zip(['x','y','z'],map(float, xcoord[1:4])))
                atom['element']=xcoord[0]
                coordat.append(atom)
                forces.append(map(float, xcoord[4:7]))
                i+=1
            line = f.readline() #ignore line
            self.freq = line.split()
            line = f.readline() # smiles
            self.smiles = line.split()
            line = f.readline() # ichi key
            self.inchi = line.split()
            self.coords['coords']=coordat
            self.add_geom(e, nat, labels,atomcoords,forces)
        er = self.energy[0]
        self.json.update(self.coords)
        self.json.update(self.prop)
        self.json['inchikeys']=self.inchi
        self.json['smiles']=self.smiles
        if  len(self.freq): self.json['freqs']=self.freq

    def add_geom(self, e, nat, labels,atomcoords,forces):
        self.energy.append(e)
        self.nats.append(nat)
        self.label.append(labels)
        self.coord.append(atomcoords)
        self.force.append(forces)


    def toscreen(self, no, grad=False, energy=False, mono_grad=False):
        i = 0
        print(self.nats[no])
# energy[0]=total energy
# energy[1]=energy mon1 free
# energy[2]=energy mon1 ghost
# energy[3]=energy mon2 free
# energy[4]=energy mon2 ghost
#       if bsse:   print ('Ebsse=%.8f' % self.bsse[no]),

        enersupermole= self.energy[5*no]+self.bsse[no]
        if energy: print ('Edft=%.8f' % (self.energy[5*no]+self.bsse[no])),
        e2body = self.energy[5*no] - self.energy[5*no+1] - self.energy[5*no+ 3] + self.bsse[no]
        if energy: print ('Edft_2body=%.8f' % e2body),
        if energy: print ('Edft_mono1=%.8f' % self.energy[5*no+2]),
        if energy: print ('Edft_mono2=%.8f' % self.energy[5*no+4]),
        print('Lattice="50.00000000       0.00000000       0.00000000       0.00000000      50.00000000       0.00000000       0.00000000       0.00000000      50.00000000"'),
        print('Properties=species:S:1:pos:R:3:force:R:3:Z:I:1:fmp2_2body:R:3:fmp2_mono1:R:3:fmp2_mono2:R:3'),
        print
        for i in range(self.nats[no]):
            print ('%s '%  self.label[no][i] ),
            for x in range(3) : print (' %.8f' %  self.coord[no][i][x] ),  
            #print(' %d '% elem2charge(self.label[no][i] )),
            if grad:
#print grad_dft
                for x in range(3):  print (' %.8f '%  ( self.force[5*no][i][x] - self.forcewbsse[no][i][x] ) ), 
            print(' %d '%  elem2charge(self.label[no][i]) ),
            if mono_grad :
                # forces from monomers isolated
                #for x in range(3):  print (' %.8f '%  self.forcemons[no][i][x] ), 
                # forces from monomers isolated with gost
                for x in range(3):  print (' %.8f '%  self.force[5*no+2][i][x] ), 
                for x in range(3):  print (' %.8f '%  self.force[5*no+4][i][x] ), 
                # forces corrected by bsse
                for x in range(3):  print (' %.8f '% ( self.force[5*no][i][x] - self.force[5*no+2][i][x] 
                                                      -  self.force[5*no+4][i][x] - self.forcewbsse[no][i][x]) ), 
            print
            #       forces_mp2_2body[countg][numat][0], forces_mp2_2body[countg][numat][1],forces_mp2_2body[countg][numat][2])             #print (numat)

    def displace(self, no, atomo, coordd, delta):
        i = 0
        newcoord = []
        for l in xrange(self.nats[no]):
           newxyz = []
           k = 0
           while k < 3 :
        #       print coordd
               tmp = self.coord[no][l][k]
               if k == coordd:
                 if l == atomo:
                    tmp = self.coord[no][l][k]+ delta
               newxyz.append(tmp)
               k +=1
           newcoord.append(newxyz)
        print(self.nats[no])
        print('geom num %d' % no)
        while i < self.nats[no]:
            print (' %s %.8f %.8f %.8f  ' %  (self.label[no][i] , newcoord[i][0] ,  newcoord[i][1],  newcoord[i][2]))
            i += 1

    def displace2file(self,file, no, atomo, coordd, delta):
        i = 0
        newcoord = []
        for l in xrange(self.nats[no]):
           newxyz = []
           k = 0
           while k < 3 :
        #       print coordd
               tmp = self.coord[no][l][k]
               if k == coordd:
                 if l == atomo:
                    tmp = self.coord[no][l][k]+ delta
               newxyz.append(tmp)
               k +=1
           newcoord.append(newxyz)
        while i < self.nats[no]:
            file.write (' %s %.8f %.8f %.8f \n' %  (self.label[no][i] , newcoord[i][0] ,  newcoord[i][1],  newcoord[i][2]))
            i += 1

    def distance(self, no, a, b):
        x = self.coord[no][a][0] - self.coord[no][b][0]
        y = self.coord[no][a][1] - self.coord[no][b][1]
        z = self.coord[no][a][2] - self.coord[no][b][2]
        return sqrt(x*x + y*y + z*z)

#    def p_save(self, fname):
#        import pickle
#        with open(fname, "wb" ) as oFile:
#            pickle.dump (self, oFile)
#
#    def p_load(self, fname):
#        import pickle
#        with open(fname, "rb" ) as iFile:
#            self = pickle.load(iFile)


def print_xyz(file, countg, natpergeom,energy_2b,atomnames,geometries,forces_mp2_2body):
        file.write ('%i\n  %.8f\n' % (natpergeom[countg],energy_2b[countg]))
        numat = 0
        while numat < natpergeom[countg]:
             file.write( ' %s  %.8f %.8f %.8f %.8f %.8f %.8f \n' % (atomnames[numat], geometries[countg][numat][0],geometries[countg][numat][1],geometries[countg][numat][2], 
                   forces_mp2_2body[countg][numat][0], forces_mp2_2body[countg][numat][1],forces_mp2_2body[countg][numat][2]) )
             #print (numat)
             numat += 1

def print_botgeom(file):
        file.write('end     \n')

def elem2charge(elem):
        if elem == "C":
           return 6
        if elem == "H":
           return 1
        if elem == "O":
           return 8
        if elem == "N":
           return 7
        if elem == "Li":
           return 3

def print_body(obj, no, file):
    file.write('%d \n ' % obj.nats[no])
    file.write('geom num %d ' % no)
    j = 0
    while j < obj.nats[no]:
        file.write('\n' )
        file.write('%s %.8f %.8f %.8f' %  (obj.label[no][j], obj.coord[no][j][0], obj.coord[no][j][1], obj.coord[no][j][2]))
        j += 1

