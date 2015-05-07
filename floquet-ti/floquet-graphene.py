import numpy as np
import matplotlib.pyplot as plt

def MatrixPlus(BandNum, kx, ky, Omega, A):
    '''
    Build the Floquet LHS matrix at momentum (kx, ky) and applied field frequency Omega.

    This matrix has tau_z = +1.

    The size of the matrix is 2*(2 * BandNum + 1) x 2*(2 * BandNum + 1). The middle entry of the matrix is n = 0, and the blocks extend out to -BandNum and BandNum.

    A is the magnitude of the vector potential.
    '''
    #declare matrix with correct size
    #must be complex
    MatrixPlus = np.zeros((2*(2*BandNum + 1), 2*(2*BandNum + 1)), dtype='complex128')

    #fill in diagonal of the matrix
    #the nth block has diagonal entries -n * Omega
    for ii in np.arange(2*BandNum + 1):
        MatrixPlus[ii, ii] = (-BandNum + ii) * Omega
        MatrixPlus[ii+1, ii+1] = (-BandNum + ii) * Omega

        MatrixPlus[ii+1, ii] = kx + 1j*ky
        MatrixPlus[ii, ii+1] = kx - 1j*ky

        MatrixPlus[ii + 3, ii] = A
        MatrixPlus[ii, ii + 3] = A

    return MatrixPlus


def MatrixMinus(BandNum, kx, ky, Omega, A):
    '''
    Build the Floquet LHS matrix at momentum (kx, ky) and applied field frequency Omega.

    This matrix has tau_z = -1.

    The size of the matrix is 2*(2 * BandNum + 1) x 2*(2 * BandNum + 1). The middle entry of the matrix is n = 0, and the blocks extend out to -BandNum and BandNum.

    A is the magnitude of the vector potential.
    '''
    #declare matrix with correct size
    #must be complex
    MatrixPlus = np.zeros((2*(2*BandNum + 1), 2*(2*BandNum + 1)), dtype='complex128')

    #fill in diagonal of the matrix
    #the nth block has diagonal entries -n * Omega
    for ii in np.arange(2*BandNum + 1):
        MatrixPlus[ii, ii] = (-BandNum + ii) * Omega
        MatrixPlus[ii+1, ii+1] = (-BandNum + ii) * Omega

        MatrixPlus[ii+1, ii] = kx + 1j*ky
        MatrixPlus[ii, ii+1] = kx - 1j*ky

        MatrixPlus[ii + 1, ii + 2] = -A
        MatrixPlus[ii + 2, ii + 1] = -A

    return MatrixPlus


def CalcBands_Plus(kVals, BandNum, Omega, A):
    '''
    Calculate the bands along a particular slice of k-space

    (kx ky)
    (kx ky)
    ...
    (kx ky)

    where each row is a particular point in k-space at which the band structure should be calculated.
    Ideally, one chooses a line along the Brillouin zone.

    BandNum is the same BandNum appearing above.

    Omega is the drive field frequency, A is the vector potential magnitude.
    '''
    energies = np.zeros((2*(2*BandNum+1), len(kVals)), dtype='complex128')
    
    for ii in np.arange(len(kVals)):
        energies[:, ii] = np.linalg.eig(MatrixPlus(BandNum, kVals[ii, 0], kVals[ii, 1], Omega, A))[0]

    return energies

def CalcBands_Minus(kVals, BandNum, Omega, A):
    '''
    Calculate the bands along a particular slice of k-space

    (kx ky)
    (kx ky)
    ...
    (kx ky)

    where each row is a particular point in k-space at which the band structure should be calculated.
    Ideally, one chooses a line along the Brillouin zone.

    BandNum is the same BandNum appearing above.

    Omega is the drive field frequency, A is the vector potential magnitude.
    '''
    energies = np.zeros((2*(2*BandNum+1), len(kVals)), dtype='complex128')
    
    for ii in np.arange(len(kVals)):
        energies[:, ii] = np.linalg.eig(MatrixMinus(BandNum, kVals[ii, 0], kVals[ii, 1], Omega, A))[0]

    return energies

BandNum = 5
kx = 11
ky = 12
Omega = 13
A = 14

print MatrixPlus(BandNum, kx, ky, Omega, A).real
#print MatrixMinus(BandNum, kx, ky, Omega, A)
