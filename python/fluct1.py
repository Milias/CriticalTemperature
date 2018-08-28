
from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<4

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Fluctuations: density'
description = """Computing excitonic and scattering densities using the
full many-body T-matrix.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

N = 1<<17

n = 3.27e23 * sys.lambda_th**3

print(fluct_mu(n, sys))

exit()

a = -1
mu_e = -1
mu_h = ideal_mu_h(mu_e, sys)
ac_max = fluct_ac(0, mu_e, mu_h, sys)

x = iter_linspace(ac_max, 10, N)

batch = job_api.new_batch(name, description)

job_api.submit(
  fluct_pmi,
  x,
  itertools.repeat(mu_e),
  itertools.repeat(mu_h),
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Computing excitonic density.'
)

job_api.process()


