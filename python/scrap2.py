from common import *

N = 1<<10
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

plt.autoscale(enable = True, axis = 'x', tight = True)

"""
lambda_s = 1e-14

x = linspace(1.1 * sys.E_1 * 4, -10, N)

y = array([wf_n_py(E, 1/lambda_s, sys) for E in x])
y_2d = array([wf_2d_n_py(E, 1/lambda_s, sys) for E in x])
#y3 = array([wf_2d_s_py(E, 1/lambda_s, sys)[-3] for E in x])
#y4 = array([wf_s_py(E, 1/lambda_s, sys)[-3] for E in x])

plt.ylim(-2, 10)

plt.axvline(x = wf_2d_E_py(1/lambda_s, sys), color = 'g', linestyle = '--')
for n in arange(5, dtype = float64):
  plt.axvline(x = sys.get_E_n(n + 0.5), color = 'b', linestyle = '--')
  plt.axvline(x = sys.get_E_n(n + 1.0), color = 'r', linestyle = '--')

plt.plot(x, y, 'r.-')
plt.plot(x, y_2d, 'b.-')
#plt.plot(x, y3)
#plt.plot(x, y4)
plt.show()

exit()
"""

n = 4e26 * sys.lambda_th**3
n_2d = 4e17 * sys.lambda_th**2

print('n: %e, n_2d: %e' % (n, n_2d))

x = linspace(1e-10, 15, N)
y = array([mb_iod(n, 1/x, sys) for x in x])
#y2 = array([mb_iod_l(n, 1/x, sys) for x in x])
y3 = array([mb_2d_iod(n, 1/x, sys) for x in x])
#y4 = array([mb_2d_iod_l(n, 1/x, sys) for x in x])

plot_type = 'semilogy'

getattr(plt, plot_type)(x, y, 'r-')
#getattr(plt, plot_type)(x, y2, 'b-')
getattr(plt, plot_type)(x, y3, 'r--')
#getattr(plt, plot_type)(x, y4, 'b--')

plt.show()

