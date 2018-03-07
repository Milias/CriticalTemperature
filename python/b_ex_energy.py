from common import *

N = 1<<10
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

print('Groundstate energy (Coulomb): %f, %f meV' % (sys.E_1, 1e3 * sys.E_1 * sys.energy_th))

print('Shift: %f' % sys.delta_E)

x = linspace(1e-10, 0.2, N)
y = array([analytic_b_ex_E(1/x, sys) for x in x])
y2 = sys.delta_E * x + sys.E_1

plt.plot(x, y, 'r-')
plt.plot(x, y2, 'b-')
plt.autoscale(enable = True, axis = 'x', tight = True)
plt.axhline(y = sys.E_1, color = 'g', linestyle = '--')
#plt.axhline(y = 0, color = 'k', linestyle = '-')
plt.show()

