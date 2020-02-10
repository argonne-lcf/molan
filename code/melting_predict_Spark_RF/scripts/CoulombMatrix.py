#!/data/ganesh/Software/anaconda/bin/python -tt    
# E-MAIL:-  gsivaraman@anl.gov
# import modules used here 
from __future__ import print_function
import sys
import pybel
from time import time
from numpy import *
from numba import jit
import json


"""
kernprof -l script.py
python -m line_profiler script.py.lprof
Uncomment @profile after identifying the bottleneck!
"""



"""
A python script to generate sorted Coloumb matrix for given chosen input molecule
Uses Numba JIT to compile NumPy function

"""


 
def largest_molecule_size(training_input) :
    '''
    All the rows of Coulomb matrix shoud be of same dimension. Hence we need number of atoms in the largest molecule . 
    This function uses Pybel to compute just that!
    '''    
    mols = [pybel.readstring("smi", molecule)  for molecule in training_input  ]
    [mol.OBMol.AddHydrogens() for mol in mols]
    return int(max([ len(mol.atoms)  for mol in mols]))



def periodicfunc(element):
    """
    A function to output atomic number for each element in the periodic table
    """
    f = open("pt.dat")
    atomicnum = [line.split()[1] for line in f if line.split()[0] == element]
    f.close()
    return int(atomicnum[0])



def process_smile(smi):    
    """
    A function to convert SMILESTRING to 3D coordinates using  openbabel
    """
    mol = pybel.readstring('smi',smi)
    mol.OBMol.AddHydrogens()
    mol.make3D()
    item = mol.write('xyz')
    index = int( item.split()[0] )
    coord = item.split()[1:]
    coordrow =[]
    chargearray = zeros((index,1))
    xyzij= zeros((index,3))
    for i in range(index):
        coordrow = coord[4*i:4*i+4] 
        chargearray[i] = periodicfunc(coordrow[0])
        xyzij[i,:] =  coordrow[1:]
    return index, chargearray, xyzij
    


def process_json(jsonfile):
    """
    This function processes the json file. Returns  index, charge, and xyz array's required by  coulombmat function
    """
    localjson = json.load(open(jsonfile))
    index = len(localjson['gaussian']['coords'])
    chargearray = zeros((index,1))
    xyzij= zeros((index,3))
    for iter in range(index):
        local_dict = localjson['gaussian']['coords'][iter]
        xyzij[iter,:] = local_dict['x'],local_dict['y'],local_dict['z']
        chargearray[iter] = periodicfunc(local_dict['element']) 
    return index, chargearray, xyzij


def process_gdb_json(mols):
    """
    This function processes the GDB json file. Returns  index, charge, and xyz array's required by  coulombmat function
    """
    localjson = mols.coords
    index = len(mols.coords['coords'])
    chargearray = zeros((index,1))
    xyzij= zeros((index,3))
    for iter in range(index):
        local_dict = localjson['coords'][iter]
        xyzij[iter,:] = local_dict['x'],local_dict['y'],local_dict['z']
        chargearray[iter] = periodicfunc(local_dict['element'])
    return index, chargearray, xyzij




def process_xyz(filename): 
    """
    This function processes the xyz file. Returns  index, charge, and xyz array's required by  coulombmat function
    """
    xyzfile=open(filename)
    xyzheader = int(xyzfile.readline())
    xyzfile.close()   
    chargearray = zeros((xyzheader,1))
    xyzij = loadtxt(file,skiprows=2,usecols=[1,2,3])
    atominfoarray = loadtxt(file,skiprows=2,dtype=str,usecols=[0])
    chargearray = [periodicfunc(symbol)  for symbol in atominfoarray]
    return xyzheader, chargearray, xyzij
    
 
'@jit'    
def coulombmat(maxnumatom,xyzheader,chargearray,xyzij):
    """
    This function takes in an xyz input file for a molecule, number of atoms in the biggest molecule  to computes the corresponding coulomb Matrix 
    """
    i=0 ; j=0    
    cij=zeros((maxnumatom,maxnumatom))
    
    for i in range(xyzheader):
        for j in range(xyzheader):
            if i == j:
                cij[i,j]=0.5*chargearray[i]**2.4   # Diagonal term described by Potential energy of isolated atom
            else:
                dist= linalg.norm(xyzij[i,:] - xyzij[j,:])              
                cij[i,j]=chargearray[i]*chargearray[j]/dist   #Pair-wise repulsion 
    return  cij



def SortMat(unsorted_mat):
    """
    Takes in a Coloumb matrix of (mxn) dimension and performs a rowwise sorting such that ||C(j,:)|| > ||C(j+1,:)||, J= 0,1,.......,(m-1)
    Finally returns a vectorized (m*n,1) Coloumb matrix .
    """   
    summation = array([sum(x**2) for x in unsorted_mat])
    sorted_mat = unsorted_mat[argsort(summation)[::-1,],:]    
    return sorted_mat.ravel() 



def CoulombMatDescriptor(maxnumatom,xyzheader,chargearray,xyzij):
    """
    A function to produce an unravelled sorted Coloumb matrix for a chosen input molecule
    """
    unsorted_mat = coulombmat(maxnumatom,xyzheader,chargearray,xyzij)
    Cijrow = SortMat(unsorted_mat)
    return Cijrow
    


# Gather our code in a main() function
def main():
    init = time()
    training_array = ['O','CC']
    maxdim = largest_molecule_size(training_array)
    print(maxdim)
    print('\n')
    for item in training_array:
        
        xyzheader, chargearray, xyzij = process_smile(item)
        print(CoulombMatDescriptor(maxdim,xyzheader,chargearray,xyzij) )
    final = time()
    print("Program Complete in {} Sec.".format(final-init))
    

    
    
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()
