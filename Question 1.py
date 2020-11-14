import sympy as sym

# Linearisation
# φ(F, x3, x4) ~ φ(F0, x30, x40)    + (dφ/dF)(F0, x30, x40)  * (F - F0)
#                                   + (dφ/dx3)(F0, x30, x40) * (x3 - x30)
#                                   + (dφ/dx4)(F0, x30, x40) * (x4 - x40)
#
# ψ(F, x3, x4) ~ ψ(F0, x30, x40)    + (dψ/dF)(F0, x30, x40)  * (F - F0)
#                                   + (dψ/dx3)(F0, x30, x40) * (x3 - x30)
#                                   + (dψ/dx4)(F0, x30, x40) * (x4 - x40)

# Define all the involved symbolic variables
# Constants
M, m, g, ell = sym.symbols('M, m, g, ell')

# System variables
x1, x2, x3, x4, F = sym.symbols('x1, x2, x3, x4, F')

# Define φ, equation (3.2a)
phi = 4 * m * ell * x4**2 * sym.sin(x3) + 4 * F - 3 * m * g * sym.sin(x3) * sym.cos(x3)     # Numerator of (3.2a)
phi /= 4 * (M + m) - 3 * m * sym.cos(x3)**2     # Denominator of (3.2a)


# Define ψ, equation (3.2b)
psi = -3 * (m * ell * x4**2 * sym.sin(x3) * sym.cos(x3) + F * sym.cos(x3) - (M + m) * g * sym.sin(x3))
psi /= ell * (4 * (M + m) - 3 * m * sym.cos(x3)**2)
#sym.pprint(psi)     # Prints equation (3.2a) in a nice readable way for a human

# Determine the partial derivatives of φ, w.r.t to F, x3, x4
phi_deriv_F = phi.diff(F)
phi_deriv_x3 = phi.diff(x3)
phi_deriv_x4 = phi.diff(x4)

# Determine the partial derivatives of ψ, w.r.t to F, x3, x4
psi_deriv_F = psi.diff(F)
psi_deriv_x3 = psi.diff(x3)
psi_deriv_x4 = psi.diff(x4)

# The equilibrium points
F0 = 0
x30 = 0
x40 = 0

# Substitute symbols with equilibrium points
phi_deriv_F_at_equlibrium = phi_deriv_F.subs([(F, F0), (x3, x30), (x4, x40)])
phi_deriv_x3_at_equlibrium = phi_deriv_x3.subs([(F, F0), (x3, x30), (x4, x40)])
phi_deriv_x4_at_equlibrium = phi_deriv_x4.subs([(F, F0), (x3, x30), (x4, x40)])

psi_deriv_F_at_equlibrium = psi_deriv_F.subs([(F, F0), (x3, x30), (x4, x40)])
psi_deriv_x3_at_equlibrium = psi_deriv_x3.subs([(F, F0), (x3, x30), (x4, x40)])
psi_deriv_x4_at_equlibrium = psi_deriv_x4.subs([(F, F0), (x3, x30), (x4, x40)])

# Print the partial derivatives
sym.pprint(phi_deriv_F_at_equlibrium)
sym.pprint(phi_deriv_x3_at_equlibrium)
sym.pprint(phi_deriv_x4_at_equlibrium)
sym.pprint(psi_deriv_F_at_equlibrium)
sym.pprint(psi_deriv_x3_at_equlibrium)
sym.pprint(psi_deriv_x4_at_equlibrium)

# Print the equations for LaTex
print(sym.latex(phi))
print(sym.latex(psi))
print(sym.latex(phi_deriv_F_at_equlibrium))
print(sym.latex(phi_deriv_x3_at_equlibrium))
print(sym.latex(phi_deriv_x4_at_equlibrium))
print(sym.latex(psi_deriv_F_at_equlibrium))
print(sym.latex(psi_deriv_x3_at_equlibrium))
print(sym.latex(psi_deriv_x4_at_equlibrium))