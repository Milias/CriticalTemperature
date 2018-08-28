from common import *

N = 1<<14
m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

plt.autoscale(enable = True, axis = 'x', tight = True)

lambda_s = 1

x = [sys.E_1 / (0 + 0.5)**2, wf_2d_E_py(1/lambda_s, sys)]
#x = sys.E_1 / (arange(3) + 0.5)**2

for E in x:
  y = array(wf_2d_s_py(E, 1/lambda_s, sys))
  N = len(y) // 3

  y = y.reshape(N, 3)

  plt.plot(y[:, 2], y[:, 0], '.-', label = E)
  plt.ylim(- 0.3 * amax(y[:N//2, 0]), 1.1*amax(y[:N//2, 0]))
  plt.xlim(0, y[-1, 2] / 2)

plt.axhline(y = 0, color = 'k')
plt.legend(loc = 0)
plt.show()

