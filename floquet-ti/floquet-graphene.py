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
		try:
			MatrixPlus[2*ii, 2*ii] = (-BandNum + ii) * Omega
		except IndexError:
			pass
		try:
			MatrixPlus[2*ii+1, 2*ii+1] = (-BandNum + ii) * Omega
		except IndexError:
			pass

		try:
			MatrixPlus[2*ii+1, 2*ii] = kx + 1j*ky
		except IndexError:
			pass
		try:
			MatrixPlus[2*ii, 2*ii+1] = kx - 1j*ky
		except IndexError:
			pass
		
		try:
			MatrixPlus[2*ii + 3, 2*ii] = A
		except IndexError:
			pass
		try:
			MatrixPlus[2*ii, 2*ii + 3] = A
		except IndexError:
			pass

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
		try:
			MatrixPlus[2*ii, 2*ii] = (-BandNum + ii) * Omega
		except IndexError:
			pass
		try:
			MatrixPlus[2*ii+1, 2*ii+1] = (-BandNum + ii) * Omega
		except IndexError:
			pass
		
		try:
			MatrixPlus[2*ii+1, 2*ii] = kx + 1j*ky
		except IndexError:
			pass
		try:
			MatrixPlus[2*ii, 2*ii+1] = kx - 1j*ky
		except IndexError:
			pass
		
		try:
			MatrixPlus[2*ii + 1, 2*ii + 2] = -A
		except IndexError:
			pass
		try:
			MatrixPlus[2*ii + 2, 2*ii + 1] = -A
		except IndexError:
			pass

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

def CalcBands_Sorted(kVals, BandNum, Omega, A):
    '''
    Combines the CalcBands_Plus and CalcBands_Minus contributions to the band structure and sorts the eigenvalues.
    '''
    return np.sort(
        np.vstack((
            CalcBands_Plus(kVals, BandNum, Omega, A),
            CalcBands_Minus(kVals, BandNum, Omega, A)
        )),
        axis = 0
    )

def TimeAvgEnergy(kVals, BandNum, Omega, A):
    '''
    This function calculates the physical energy associated with a particular quasienergy band. Floquet theory gives the time-averaged energy E_\alpha for the quasienergy band \alpha as
    E_\alpha = \epsilon_\alpha (\omega) - \frac{\partial \epsilon_\alpha}{\partial \omega},
    where \omega is the drive frequency.
    '''
    from scipy.misc import derivative
    
    def quasienergy(omega):
        return CalcBands_Sorted(kVals, BandNum, omega, A)

    return quasienergy(Omega) - Omega * derivative(quasienergy, Omega, dx=1e-9)
    







'''
Notes on graphene high symmetry k points
'''
#Graphene lattice constant a = 1.42 A
#Set a = 1 for convenience
a = 1.

#The following points KDirac1 and KDirac2 correspond to the usual K and K'
#These are on the RHS of the BZ--can be rotated to any part of BZ
KDirac1 = np.array([[2.*np.pi/3./a, 2.*np.pi/3./a/np.sqrt(3)]])
KDirac2 = np.array([[2.*np.pi/3./a, -2.*np.pi/3./a/np.sqrt(3)]])

#gamma point at origin of BZ
Gamma = np.array([[0., 0.]])

##################################################################
##################################################################
##################################################################

#example parameters
BandNum = 15
kx = 11
ky = 12
Omega= 1.
A = .3

#test matrix calculators
# print MatrixPlus(BandNum, kx, ky, Omega, A)
# print MatrixMinus(BandNum, kx, ky, Omega, A)

#number of points to calculate band structure at in k space
NumPts_kSpace = 250

'''
Form line in k-space along which the spectrum should be evaluated
'''
#x vals
#kVals_x = np.linspace(Gamma[0,0], KDirac1[0,0], num=NumPts_kSpace).reshape((NumPts_kSpace, 1))
#y vals
#kVals_y = np.linspace(Gamma[0,1], KDirac1[0,1], num=NumPts_kSpace).reshape((NumPts_kSpace, 1))

#### Here is another choice (encloses dirac pt.)
#x vals
#kVals_x = np.linspace(KDirac1[0,0] - KDirac1[0,0]/2., KDirac1[0,0] + KDirac1[0,0]/2., num=NumPts_kSpace).reshape((NumPts_kSpace, 1))
#y vals
#kVals_y = np.linspace(KDirac1[0,1] - KDirac1[0,1]/2., KDirac1[0,1] + KDirac1[0,1]/2., num=NumPts_kSpace).reshape((NumPts_kSpace, 1))

#### Here is yet another choice (goes from one dirac point to the other
#x vals
kVals_x = np.linspace(-KDirac1[0,0], KDirac1[0,0], num=NumPts_kSpace).reshape((NumPts_kSpace, 1))
#y vals
kVals_y = np.linspace(-KDirac1[0,1], KDirac1[0,1], num=NumPts_kSpace).reshape((NumPts_kSpace, 1))

#stack x and y kVals arrays to form the full kVals array
kVals = np.hstack((kVals_x, kVals_y))

#test kVals
#print kVals


'''
calculate bands along kVals in k-space
'''

'''
#calculate (+) bands
#Note: Take real part -- there will be a small imaginary part due to floating pt. calcs. However, be careful -- should add a way to make sure the imaginary part is small, since if it's large something went wrong (should be diagonalizing a hermitian operator)
BandsPlus = CalcBands_Plus(kVals, BandNum, Omega, A).real

#test CalcBands_Plus
# print BandsPlus
# print BandsPlus.shape

BandsMinus = CalcBands_Minus(kVals, BandNum, Omega, A).real
'''

'''
#plot each slice in k space
for ii in np.arange(2*(2*BandNum+1)):
	plt.scatter(kVals[:,0],BandsPlus[ii, :])
        plt.scatter(kVals[:,0],BandsMinus[ii, :])

plt.xlim(kVals[0,0], -kVals[0,0])
plt.ylim(-Omega/2., Omega/2.)
plt.show()
'''

EBands = TimeAvgEnergy(kVals, BandNum, Omega, A)
#EBands = CalcBands_Sorted(kVals, BandNum, Omega, A)
for ii in np.arange(2*2*(2*BandNum+1)):
    plt.scatter(kVals[:,0], EBands[ii, :])

plt.show()
print EBands.shape
