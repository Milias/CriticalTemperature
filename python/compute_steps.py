from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<3

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Numerical steps'
description = """Computing and saving each step of the numerical solution
to the equation of state and self-consistency for the scattering length.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

N, M = 1<<5, 1<<4

n = 8.97e23 * sys.lambda_th**3
factor = 1.25

a_v =  linspace(-10, 10, N)
mu_e_v = array([analytic_mu_init_mu(n, a, sys) for a in a_v])
f = repeat(linspace(1, factor, M).reshape(1, M), N, 0)

mu_e = (f * repeat(mu_e_v.reshape(N, 1), M, 1)).flatten()
a = repeat(a_v, M)

params = stack((mu_e, a)).T

batch = job_api.new_batch(name, description)

job_api.submit(
  analytic_mu_follow,
  itertools.repeat(n, N*M),
  params,
  itertools.repeat(sys, N*M),
  size = N*M,
  block_size = bs,
  desc = 'Computing chemical potential neglecting fluctuations.'
)

job_api.process()

