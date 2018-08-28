from common import *

N = 1<<7
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

E_gs_2d = wf_2d_E_cou_py(sys)
E_gs_lim_2d = wf_2d_E_lim_py(1e6, sys)

print('Groundstate energy (Coulomb 3D): %f, %f meV' % (sys.E_1, 1e3 * sys.E_1 * sys.energy_th))
print('Groundstate energy (Coulomb 2D): %f, %f meV' % (E_gs_2d, 1e3 * E_gs_2d * sys.energy_th))
print('Groundstate energy (Limit 2D): %f, %f meV' % (E_gs_lim_2d, 1e3 * E_gs_lim_2d * sys.energy_th))

x = linspace(1e-6, 50, N)
#x = logspace(-6, log10(170), N)
y = array([wf_E_py(1/x, sys) for x in x])
y3 = array([wf_2d_E_py(1/x, sys) for x in x])
y4 = array([wf_2d_E_lim_py(1/x, sys) for x in x])

y /= -nanmin(y)
y3 /= -nanmin(y3)
y4 /= -nanmin(y4)

plot_type = 'plot'
getattr(plt, plot_type)(x, y, 'r.-')
getattr(plt, plot_type)(x, y3, 'g.-')
getattr(plt, plot_type)(x, y4, 'm.-')

#plt.axhline(y = sys.E_1, color = 'r', linestyle = '--')
#plt.axhline(y = E_gs_2d, color = 'g', linestyle = '--')
plt.axhline(y = 0, color = 'k', linestyle = '-')

#plt.ylim(E_gs_2d, 100)
plt.autoscale(enable = True, axis = 'x', tight = True)
plt.show()

