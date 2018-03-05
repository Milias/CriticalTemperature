
from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<0

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Fluctuations: density'
description = """Computing the density considering
fluctuations.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

N = 1<<10
n = 6e23 * sys.lambda_th**3
mu_e = ideal_mu_v(-1, sys)
mu_h = ideal_mu_h(mu_e, sys)
ac_max = fluct_ac(0, mu_e, mu_h, sys)
ac_pp0 = fluct_pp0_a(mu_e, mu_h, sys)
ac_pp0 = ac_pp0 if not isnan(ac_pp0) else 0

x = iter_linspace(-10, ac_pp0, N)

print((mu_e, mu_h))
print((ac_pp0, ac_max))

batch = job_api.new_batch(name, description)

job_api.submit(
  fluct_n_ex,
  x,
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Computing excitonic density contribution with fluctuations.'
)

job_api.submit(
  fluct_n_sc,
  x,
  itertools.repeat(mu_e, N),
  itertools.repeat(mu_h, N),
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Computing scattering density contribution with fluctuations.'
)

job_api.process()

