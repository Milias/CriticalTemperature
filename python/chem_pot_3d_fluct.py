from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<0
N = 1<<5

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Chemical potential (fluct) 3D'
description = """Computing the chemical potential as a function of the
density, in 3D.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print('lambda_th: %f nm, energy_th: %f meV' % (sys.lambda_th * 1e9, sys.energy_th * 1e3))

n0, n1 = 5, 25
print('n0: %e, n1: %e' % (n0, n1))
n = iter_linspace(n0, n1, N)

batch = job_api.new_batch(name, description)

mu_a = job_api.submit(
  fluct_mu,
  n,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Full chemical potential in 3D.'
)

job_api.process()

mu_a_arr = array(mu_a.result)

mu_h = job_api.submit(
  ideal_mu_h,
  mu_a_arr[:, 0],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Chemical potential for holes.'
)

job_api.process()

mu_h_arr = array(mu_h.result)
mu_t = mu_a_arr[:, 0] + mu_h_arr

n_id = job_api.submit(
  ideal_n,
  mu_t,
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 3D.'
)

n_ex = job_api.submit(
  fluct_n_ex,
  mu_t,
  mu_a_arr[:, 1],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 3D.'
)

n_sc = job_api.submit(
  fluct_n_sc,
  mu_t,
  mu_a_arr[:, 1],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 3D.'
)

job_api.process()

ls_id = job_api.submit(
  ideal_ls,
  n_id.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Screening length in 3D.'
)

job_api.process()

b_ex = job_api.submit(
  wf_E_py,
  ls_id.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Exciton binding energy in 3D.'
)

job_api.process()

