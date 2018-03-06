from common import *

N = 1<<10
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

print('Groundstate energy (Coulomb): %f, %f meV' % (sys.E_1, 1e3 * sys.E_1 * sys.energy_th))

E_shift = sqrt(32 * pi) * (sys.c_aEM / sys.eps_r)**1.5 * (sys.m_pT / 8)**0.75

print('Shift: %f' % E_shift)

x = logspace(-3, 0, N)
y = array([analytic_b_ex_E(1/x, sys) for x in x]) - sys.E_1
y2 = E_shift * x

plt.semilogx(x, y, 'r-')
plt.semilogx(x, y2, 'b-')
plt.autoscale(enable = True, axis = 'x', tight = True)
plt.axhline(y = -sys.E_1, color = 'g', linestyle = '--')
plt.axhline(y = 0, color = 'k', linestyle = '-')
plt.show()

