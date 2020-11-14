import sympy as sym

# LINEARISATION
# φ(F, x3, x4) ~ φ(F0, x30, x40)    + (dφ/dF)(F0, x30, x40)  * (F - F0)
#                                   + (dφ/dx3)(F0, x30, x40) * (x3 - x30)
#                                   + (dφ/dx4)(F0, x30, x40) * (x4 - x40)
#
# ψ(F, x3, x4) ~ ψ(F0, x30, x40)    + (dψ/dF)(F0, x30, x40)  * (F - F0)
#                                   + (dψ/dx3)(F0, x30, x40) * (x3 - x30)
#                                   + (dψ/dx4)(F0, x30, x40) * (x4 - x40)

# Define all the involved symbolic variables
# Constants
M, m, g, ell = sym.symbols('M, m, g, ell', real=True, positive=True)

# System variables
x1, x2, x3, x4, F = sym.symbols('x1, x2, x3, x4, F')

# Define φ, equation (3.2a)
phi = 4 * m * ell * x4**2 * sym.sin(x3) + 4 * F - 3 * m * g * sym.sin(x3) * sym.cos(x3)     # Numerator of (3.2a)
phi /= 4 * (M + m) - 3 * m * sym.cos(x3)**2     # Denominator of (3.2a)

# Define ψ, equation (3.2b)
psi = -3 * (m * ell * x4**2 * sym.sin(x3) * sym.cos(x3) + F * sym.cos(x3) - (M + m) * g * sym.sin(x3))      # Numerator of (3.2b)
psi /= ell * (4 * (M + m) - 3 * m * sym.cos(x3)**2)     # Denominator of (3.2b)

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

# x2' = aF - bx3
a = phi_deriv_F_at_equlibrium
b = -phi_deriv_x3_at_equlibrium

# x4' = -cF + dx3
c = -psi_deriv_F_at_equlibrium
d = psi_deriv_x3_at_equlibrium
#c = 3 / ell / (4*M+m)
#d = 3 * (M + m) * g / ell / (4*M+m)

#M_value = 0.3
#m_value = 0.1
#ell_value = 0.35
#g_value = 9.81

# Substitute symbols with values
#a_value = a.subs([(M, M_value), (m, m_value)])
#b_value = b.subs([(M, M_value), (m, m_value), (g, g_value)])
#c_value = c.subs([(M, M_value), (m, m_value), (g, g_value), (ell, ell_value)])
#d_value = d.subs([(M, M_value), (m, m_value), (g, g_value), (ell, ell_value)])

#sym.pprint(a_value)
#sym.pprint(b_value)
#sym.pprint(c_value)
#sym.pprint(d_value)

# Define the symbols: s, t and w (real)  a, b, c, d (real and positive)
s, t = sym.symbols('s, t')
w = sym.symbols('w', real=True) # w = omega
a, b, c, d = sym.symbols('a, b, c, d', real=True, positive=True)

# Define G_theta and G_x symbolically
G_theta = - c / (s**2 - d)
G_x = ((a * s**2) - (a * d) + (b * c)) / (s**4 - (d * s**2))

# Impulse, Step and Frequency response of G_theta and G_x
# Push/Step: F_s = 1 / s     Shake/Frequency: F_s = w / (s**2 + w**2)   Kick (Dirac pulse)/Impulse: F_s = 1
F_s_impulse = 1
F_s_step = 1 / s
F_s_frequency = w / (s**2 + w**2)

X3_s_impulse_G_theta = G_theta * F_s_impulse
x3_t_impulse_G_theta = sym.inverse_laplace_transform(X3_s_impulse_G_theta, s, t)

X3_s_step_G_theta = G_theta * F_s_step
x3_t_step_G_theta = sym.inverse_laplace_transform(X3_s_step_G_theta, s, t)

X3_s_frequency_G_theta = G_theta * F_s_frequency
x3_t_frequency_G_theta = sym.inverse_laplace_transform(X3_s_frequency_G_theta, s, t, w)

X1_s_impulse_G_x = G_x * F_s_impulse
x1_t_impulse_G_x = sym.inverse_laplace_transform(X1_s_impulse_G_x, s, t)

X1_s_step_G_x = G_x * F_s_step
x1_t_step_G_x = sym.inverse_laplace_transform(X1_s_step_G_x, s, t)

X1_s_frequency_G_x = G_x * F_s_frequency
x1_t_frequency_G_x = sym.inverse_laplace_transform(X1_s_frequency_G_x, s, t, w)

# Print the symbolic expressions
#sym.pprint(x3_t_impulse_G_theta.simplify())
#sym.pprint(x3_t_step_G_theta.simplify())
sym.pprint(x3_t_frequency_G_theta.simplify())
#sym.pprint(x1_t_impulse_G_x.simplify())
#sym.pprint(x1_t_step_G_x.simplify())
#sym.pprint(x1_t_frequency_G_x.simplify())

# Print the symbolic expressions for LaTex
#print(sym.latex(x3_t_impulse_G_theta.simplify()))
#print(sym.latex(x3_t_step_G_theta.simplify()))
#print(sym.latex(x3_t_frequency_G_theta.simplify()))
#print(sym.latex(x1_t_impulse_G_x.simplify()))
#print(sym.latex(x1_t_step_G_x.simplify()))
#print(sym.latex(x1_t_frequency_G_x.simplify()))