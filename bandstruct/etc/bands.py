from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

# for zincblende band structure

sigx = np.array([ [0, 1], [1, 0] ])
sigy = np.array([ [0, -1j], [1j, 0] ])
sigz = np.array([ [1, 0], [0, -1] ])



A = 100
M = 1
B = 1

bound = np.pi


def d_x(kx, ky):
	return A*np.sin(kx)

	
def d_y(kx, ky):
	return A*np.sin(ky)

	
def d_z(kx, ky):
	return M - 4*B + 2*B*( np.cos(kx) + np.cos(ky) )

	
def ham(kx, ky):
	return d_x(kx, ky)*sigx + d_y(kx, ky)*sigy + d_z(kx, ky)*sigz
	
	
def band1(kx, ky):
	return np.linalg.eig( ham(kx, ky) )[0][0]
	
	
def band2(kx, ky):
	return np.linalg.eig( ham(kx, ky) )[0][1]
	




#generate z vector
X = np.arange(-bound, bound, 0.25)
l1 = len(X)
Y = np.arange(-bound, bound, 0.25)
l2 = len(Y)
X, Y = np.meshgrid(X, Y)	

Z1 = np.zeros([l1, l2])
Z2 = np.zeros([l1, l2])

for ii in np.arange(0, l1, 1):
	for jj in np.arange(0, l2, 1):
		Z1[ii, jj] = band1(X[ii,jj], Y[ii,jj]).real
		Z2[ii, jj] = band2(X[ii,jj], Y[ii,jj]).real
	

print X.shape


fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.savefig("bands.pdf", fmt="pdf")

