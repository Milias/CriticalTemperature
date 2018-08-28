from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<4
N = 1<<15

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1 # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print((sys.m_e, sys.m_h))

mu_e = ideal_2d_mu_v(30, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)
mu_t = mu_e + mu_h

mu_e, mu_h = mu_e * sys.energy_th, mu_h * sys.energy_th

k1, k2 = 1e-3, 1e-2
w, th = 1e-1, linspace(0, 2*pi, N)
k = sqrt(k1 * k1 + k2 * k2 - 2 * k1 * k2 * cos(th))

ids = (1, 1, 1, 1)
z = -1j
dk, dw = 1e-1, 1e-3
elem_v = plasmon_potcoef(1, 1e-2, 1e-3, mu_e, mu_h, sys)
elem = elem_v[0] + 1j * elem_v[1]

print(elem)

result = plasmon_sysmatelem(*ids, z, dk, elem, sys)

print(result)

batch = job_api.new_batch(name, description)

green = job_api.submit(
  plasmon_green,
  itertools.repeat(w, N),
  k,
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Plasmon Green function.'
)

job_api.process()

