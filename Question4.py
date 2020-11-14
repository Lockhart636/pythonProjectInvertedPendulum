import sympy as sym
import control as ctrl
import matplotlib.pyplot as plt
import numpy as np

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

# The parameters of question 4
M_value = 0.3
m_value = 0.1
ell_value = 0.35
g_value = 9.81

# Substitute the symbols with the parameters
a_value = float(a.subs([(M, M_value), (m, m_value)]))
b_value = float(b.subs([(M, M_value), (m, m_value), (g, g_value)]))
c_value = float(c.subs([(M, M_value), (m, m_value), (g, g_value), (ell, ell_value)]))
d_value = float(d.subs([(M, M_value), (m, m_value), (g, g_value), (ell, ell_value)]))

# Resolution and time
n_points = 500  # Resolution
t_final = 0.2   # t ∈ [0, 0.2] s
t_span = np.linspace(0, t_final, n_points)      # Array of time instants

# Input signal
input_signal = np.sin(100 * t_span**2)

# Transfer function
G_theta = ctrl.TransferFunction([-c_value], [1, 0, -d_value])   # -c/(s^2 - d)
G_x = ctrl.TransferFunction([a_value, 0, (-a_value * d_value) + (b_value * c_value)], [1, 0, -d_value, 0, 0])   # (as^2 - ad + bc)/(s^4 - ds^2)

#t_out_G_theta, y_out_G_theta, x_out_G_theta = ctrl.forced_response(G_theta, t_span, input_signal)
#t_out_G_x, y_out_G_x, x_out_G_x = ctrl.forced_response(G_x, t_span, input_signal)

# Response of the system
t_out_G_theta, y_out_G_theta, x_out_G_theta = ctrl.forced_response(G_theta, t_span, input_signal)
t_out_G_x, y_out_G_x, x_out_G_x = ctrl.forced_response(G_x, t_span, input_signal)

#plt.plot(t_out_G_theta, y_out_G_theta)
#plt.show()

#plt.plot(t_out_G_x, y_out_G_x)
#plt.show()
# Plot the graphs
plt.plot(t_out_G_theta, y_out_G_theta)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Angle of the Rod (rad)')
plt.savefig("Question4ThetaAgainstTime.eps", format = "eps")
plt.show()

plt.plot(t_out_G_theta, y_out_G_x)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Horizontal Position of the Cart (m)')
plt.savefig("Question4XAgainstTime.eps", format = "eps")
plt.show()

#t_imp, theta_imp = ctrl.impulse_response(G_theta)   # Kick
#plt.plot(t_imp, theta_imp)
#plt.show()

#t_step, theta_step = ctrl.step_response(G_theta)    # Push
#plt.plot(t_step, theta_step)
#plt.show()
