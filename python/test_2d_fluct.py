from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<4
N = 1<<12

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 30 # K
sys = system_data(m_e, m_h, eps_r, T)

print((sys.m_pe, sys.m_ph))

mu_e = 1
mu_h = ideal_2d_mu_h(mu_e, sys)
E = 4

print((mu_e, mu_h, mu_e + mu_h))

x = linspace(0, 1e3, N)

y = array([fluct_2d_I2(z, E, mu_e, mu_h, sys) for z in x])

plt.plot(x, real(y), 'r-')
plt.plot(x, imag(y), 'b-')
plt.show()

exit()

name = 'Testing 2D'
description = """Testing 2D analytic densities.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

mu_t = -1
mu_e = ideal_2d_mu_v(mu_t, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)

print('mu_e: %f, mu_h: %f' % (mu_e, mu_h))

chi_ex = iter_linspace(mu_t + global_eps, -global_eps, N)

batch = job_api.new_batch(name, description)

n_id = job_api.submit(
  ideal_2d_n,
  itertools.repeat(mu_e, N),
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 2D.'
)

n_ex = job_api.submit(
  analytic_2d_n_ex,
  itertools.repeat(mu_t, N),
  chi_ex,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 2D.'
)

n_sc = job_api.submit(
  analytic_2d_n_sc,
  itertools.repeat(mu_t, N),
  chi_ex,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 2D.'
)

job_api.process()

