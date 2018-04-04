from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<0
N = 1<<14

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Testing 2D'
description = """Testing 2D analytic densities.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

mu_e = -0.1
mu_h = ideal_2d_mu_h(mu_e, sys)

mu_t = mu_e + mu_h

a_max = sqrt(-mu_t * pi / 2)

a = iter_linspace(0, 2 * a_max, N)

batch = job_api.new_batch(name, description)

job_api.submit(
  ideal_2d_n,
  itertools.repeat(mu_e, N),
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 2D.'
)

job_api.submit(
  analytic_2d_n_ex,
  itertools.repeat(mu_t, N),
  a,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 2D.'
)

job_api.submit(
  analytic_2d_n_sc,
  itertools.repeat(mu_t, N),
  a,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 2D.'
)

job_api.process()

