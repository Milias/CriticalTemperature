from common import *

N = 1<<10
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

E_gs_2d = wf_2d_E_cou_py(sys)

print('Groundstate energy (Coulomb): %f, %f meV' % (sys.E_1, 1e3 * sys.E_1 * sys.energy_th))
print('Groundstate energy (Coulomb 2D): %f, %f meV' % (E_gs_2d, 1e3 * E_gs_2d * sys.energy_th))

x = linspace(1e-10, 50, N)
y = array([wf_E_py(1/x, sys) for x in x])
y2 = sys.delta_E * x + sys.E_1
y3 = array([wf_2d_E_py(1/x, sys) for x in x])
y4 = sys.delta_E * x + E_gs_2d

#plt.plot(x, y, 'r-')
#plt.plot(x, y2, 'b-')
#plt.plot(x, y3, 'g-')
plt.semilogy(x, -y3, 'g-')
#plt.plot(x, y4, 'm-')
plt.autoscale(enable = True, axis = 'x', tight = True)
#plt.axhline(y = sys.E_1, color = 'b', linestyle = '--')
#plt.axhline(y = E_gs_2d, color = 'm', linestyle = '--')
#plt.axhline(y = 0, color = 'k', linestyle = '-')
#plt.ylim(E_gs_2d, 100)
plt.show()

