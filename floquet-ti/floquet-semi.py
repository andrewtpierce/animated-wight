import numpy as np
from scipy.integrate import complex_ode

sigx = np.array([ [0, 1], [1, 0] ])
sigy = np.array([ [0, -1j], [1j, 0] ])
sigz = np.array([ [1, 0], [0, -1] ])

V = np.array([0, 1, 0])

#eps_k = 0
A = 1
M = 1
B = 1
w = 1
#odesolve params
y0 = [1, 1]	
t_output = np.arange(0, 5, 0.1)

def d_x(kx, ky):
	return A*np.sin(kx)

def d_y(kx, ky):
	return A*np.sin(ky)
	
def d_z(kx, ky):
	return M - 4*B + 2*B*( np.cos(kx) + np.cos(ky) )
	
def ham(kx, ky, t):
	return d_x(kx, ky)*sigx + d_y(kx, ky)*sigy + d_z(kx, ky)*sigz + np.cos(w*t) * (V[0]*sigx + V[1]*sigy + V[2]*sigz)
	

'''
define function to be integrated
psi_k
'''
def sol(kx, ky):
	def ham_k(t):
		return ham(kx, ky, t)
	def f(y_vec, t):
		y_1, y_2 = y_vec
		fun = -1j * np.dot( ham_k(t), np.array([ [y_1],[y_2] ]) )
		return [fun[0,0], fun[1,0]]

	y_result = complex_ode(f, y0, t_output)
	
	return np.array([y_result.real, y_result.imag])

print sol(.1, .1)