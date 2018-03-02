
from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<0

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Fluctuations: chemical potential'
description = """Computing the chemical potential considering
fluctuations.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

N = 1<<5

x0, x1 = 1e23, 1e24
x = iter_linspace(x0 * sys.lambda_th**3, x1 * sys.lambda_th**3, N, func = iter_log_func)

batch = job_api.new_batch(name, description)

job_api.submit(
  fluct_mu,
  x,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Computing chemical potential with fluctuations.'
)

job_api.process()


