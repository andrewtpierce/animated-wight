import numpy as np

def H_Ribbon_Zigzag(NumCells, kx):
    '''
    Calculates hamiltonian matrix for a graphene ribbon.
    The ribbon is finite in the y direction and infinite in the x direction. NumCells in the number of ``honeycomb'' cells in the y direction.

    kx is a good quantum number, so a 1-D band structure can be calculated.
    '''
    # number of unique lattice sites in finite direction
    NumLatticeSites = 2 * (NumCells + 1)
    hamiltonian = np.zeros((NumLatticeSites, NumLatticeSites), dtype= 'complex128')

    '''
    Each site within the finite-direction repeating unit is coupled to 
    '''
