from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K
sys = system_data(m_e, m_h, eps_r, T)

E = 1
n = 4e23 * sys.lambda_th**3
a = -10

print(fluct_bmi(a, -10, -10, sys))
exit()

N = 1<<4
x = linspace(0, 1000, N)
y = [fluct_bfi(E, a, ideal_mu(n, sys.m_pe), ideal_mu(n, sys.m_ph), sys) for E in x]

plt.plot(x, y)
plt.show()

