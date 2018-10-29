from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<3
N_k = 1<<12
N_w = 1<<13

N_total = N_k * N_w

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1 # K
#m_e, m_h, eps_r, T = 1, 1, 6.56, 1 # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = r"""Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (
  sys.m_e, sys.m_h, sys.eps_r, sys.T
)

print((sys.m_e, sys.m_h))
mu_e, v_1 = 1, 1e2
mu_h = sys.m_eh * mu_e

k0, k1 = 10, 10
kvec = sqrt(k0**2 + k1**2 - 2*k0*k1*cos(linspace(0, pi, N_k)))
kmin, kmax = amin(kvec), amax(kvec)
wpl0 = plasmon_disp(kmin, mu_e, mu_h, v_1, sys)
wpl1 = plasmon_disp(kmax, mu_e, mu_h, v_1, sys)

print('%f, %f' % (kmin, kmax))

wimgzero = sys.m_pe * kmax**2 + 2 * sqrt(mu_e * sys.m_pe) * kmax

w0 = 3 * wpl1

if isnan(w0):
  w0 = 3 * wimgzero

print(w0)

wvec = linspace(0, w0, N_w)
y = array([plasmon_potcoef([w, k0, k1], mu_e, mu_h, v_1, sys, 1e-2) for w in wvec])
y_static = plasmon_potcoef([0, k0, k1], mu_e, mu_h, v_1, sys, 1e-2)

"""
plt.plot(wvec, y[:, 0], 'r-')
plt.plot(wvec, y[:, 1], 'b-')

plt.axhline(y = y_static[0], color = 'r', linestyle = '--')
plt.axhline(y = y_static[1], color = 'b', linestyle = '--')

plt.axvline(x = wpl0, color = 'g')
plt.axvline(x = -wpl0, color = 'g')
plt.axvline(x = wpl1, color = 'm')
plt.axvline(x = -wpl1, color = 'm')
"""

#"""
x = linspace(0, 1, N_w)
plt.scatter(y[:,0], y[:,1], marker = '.', c = x)
plt.plot(*y_static, 'ko')
plt.axis('equal')
#"""

#plt.ylim(-1, 1)

plt.show()

exit()

w0, k1, k2 = 2e-2, 3e-3, 1e-5
w = linspace(-w0, w0, N_w)

batch = job_api.new_batch(name, description)

green = job_api.submit(
  plasmon_potcoef,
  w,
  itertools.repeat(k1, N_total),
  itertools.repeat(k2, N_total),
  itertools.repeat(mu_e, N_total),
  itertools.repeat(mu_h, N_total),
  itertools.repeat(sys, N_total),
  size = N_total,
  block_size = bs,
  desc = 'Plasmon: Potential matrix elements.'
)

job_api.process()

