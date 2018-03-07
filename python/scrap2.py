from common import *

N = 1<<10
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

n = 4e28 * sys.lambda_th**3

x = linspace(1e-10, 0.2, N)
y = array([analytic_iod_mb(n, 1/x, sys) for x in x])
y2 = array([analytic_iod_mb_l(n, 1/x, sys) for x in x])

y = 1 - y
y2 = 1 - y2

plot_type = 'semilogy'

getattr(plt, plot_type)(x, y, 'r-')
getattr(plt, plot_type)(x, y2, 'b-')
plt.autoscale(enable = True, axis = 'x', tight = True)
plt.show()

