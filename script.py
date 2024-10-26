import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize, linprog
import matplotlib.pyplot as plt

t0, tf = 1, 10
T = np.linspace(t0, tf, 1000)

ORDER = 2
FCT = np.sinc

# Input temporal data
temporal_data = list()

# Initializing with constant value.
temporal_data.append(np.ones_like(T))

#
# Adding polynomial terms.
for i in range(ORDER - 1):
	temporal_data.append(np.multiply(
		temporal_data[-1], T
	))

#
# Adding base function.
temporal_data2 = list()
temporal_data2.append(FCT(T))

#
# Adding integrals.
for i in range(ORDER):
	temporal_data2.append(cumulative_trapezoid(
		temporal_data2[-1], T, initial=0
	))

#
# Adding the function data in reverse order.
temporal_data.extend(reversed(temporal_data2))

# Finite elements A matrix. The base function goes at
# the end so that we can just append the 1 in the cost
# function.
# A = np.transpose(np.vstack([Cst, X, F2, F1, F0]))
A = np.transpose(np.vstack(temporal_data))

#
# Generic minimize, fails often
def cost(sol):
	full_sol = np.append(sol, 1.)
	return np.linalg.norm(np.matmul(A, full_sol))
	
optimize_result = minimize(cost, np.zeros(2*ORDER))
print(f"OPTIMIZE = {optimize_result}")
coef = optimize_result.x
#
# We only keep the homogeneous coefficients.
hcoef = coef[ORDER:]

def solve_initial(coefs, t0):
	pcoefs, hcoefs = coef[:ORDER], coef[ORDER:]
	# print("hcoefs=")
	# print(hcoefs)
	T0 = np.array([
		t0 ** i for i in range(ORDER)
	])
	# print("T0=")
	# print(T0)
	#
	# Matrix of the remaining hcoefs at initial time.
	A = np.array([
		[
			hcoefs[ORDER - 1 - (i - j - 1)] if j < i else int(i==j)
			for j in range(ORDER)
		]
		for i in range(ORDER)
	])
	# print("A=")
	# print(A)
	#
	# Matrix of the polynomial coefficients still present at
	# initial time.
	B = np.array([
		[
			pcoefs[i+j] if i + j < ORDER else 0
			for j in range(ORDER)
		]
		for i in range(ORDER)
	])
	# print("B=")
	# print(B)
	#
	# the minus sign is because we are moving the polynoms to the right-hand
	# side for this resolution.
	B0 = -np.matmul(B, T0)
	# print("B0=")
	# print(B0)
	#
	# We solve Ax?=BxT0 to get the initial values of the derivatives
	# for the function.
	return np.linalg.solve(A, B0)

init = solve_initial(coef, T[0])

def apostrophe(n):
	return "'" * n

def pretty_equation(coefs, inits, t0):
	coefs2 = list(coefs) + [1.]
	rtext = [
		f"{coef} x f{apostrophe(i)}"
		for i, coef in enumerate(coefs2)
	]
	itext = [
		f"f{apostrophe(i)}({t0}) = {init}"
		for i, init in enumerate(inits)
	]
	return " + ".join(reversed(rtext)) + " ~= 0\n" + "\n".join(itext)

print("=" * 80)
print(pretty_equation(hcoef, init, T[0]))
print("=" * 80)

#
# We built the verification matrix.
V = np.vstack([
	np.array([
		int(j == i+1)
		for j in range(ORDER)
	])
	for i in range(ORDER-1)
] + [-hcoef])

def deriv(t, F):
	return np.matmul(V, F)

#
# We solve the differential equation to see how much error we have.
V0 = init
F0 = FCT(T)
#
# We only keep the first line because we aren't interested in the derivative.
verif = solve_ivp(deriv, (t0, tf), V0, t_eval=T).y[0]

error = np.linalg.norm(F0 - verif)
print("Error =", error)

plt.plot(T, F0, label="function")
plt.plot(T, verif, label="verif")
plt.legend()
plt.savefig('fig.png')
