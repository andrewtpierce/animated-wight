from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

# for graphene
# RMP 81 109

sigx = np.array([ [0, 1], [1, 0] ])
sigy = np.array([ [0, -1j], [1j, 0] ])
sigz = np.array([ [1, 0], [0, -1] ])



A = 100
M = 1
B = 1


#graphene parameters
t = 1
tp = -.2*t

bound = np.pi

def f_g(kx, ky):
	return 2*np.cos( np.sqrt(3)*ky ) + 4*np.cos( np.sqrt(3)/2.*ky )*np.cos( 3./2.*kx )

def band1(kx, ky):
	return t*np.sqrt(3 + f_g(kx, ky) ) - tp*f_g(kx, ky)
	
	
def band2(kx, ky):
	return -t*np.sqrt(3 + f_g(kx, ky) ) - tp*f_g(kx, ky)
	




#generate z vector
X = np.arange(-bound, bound, 0.1)
l1 = len(X)
Y = np.arange(-bound, bound, 0.1)
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

plt.savefig("bands_graphene.pdf", fmt="pdf")
#plt.show()
