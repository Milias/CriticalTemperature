from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<2
N = 1<<12

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

name = 'Chemical potential (analytic) 2D'
description = """Computing the chemical potential as a function of the
density, in 2D.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print('lambda_th: %f nm, energy_th: %f meV' % (sys.lambda_th * 1e9, sys.energy_th * 1e3))

print(analytic_2d_mu(1e5, sys))
exit()

n0, n1 = 1e-2, 5
print('n0: %e, n1: %e' % (n0, n1))
n = iter_linspace(n0, n1, N)

batch = job_api.new_batch(name, description)

mu_ls = job_api.submit(
  analytic_2d_mu,
  n,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Analytic chemical potential in 2D.'
)

job_api.process()

mu_ls_arr = array(mu_ls.result)

mu_h = job_api.submit(
  ideal_2d_mu_h,
  mu_ls_arr[:, 0],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Chemical potential for holes.'
)

chi_ex = job_api.submit(
  wf_2d_E_py,
  mu_ls_arr[:, 1],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Binding energy of excitons.'
)

job_api.process()

mu_h_arr = array(mu_h.result)
mu_t = mu_ls_arr[:, 0] + mu_h_arr

job_api.submit(
  ideal_2d_n,
  mu_t,
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 2D.'
)

job_api.submit(
  analytic_2d_n_ex,
  mu_t,
  chi_ex.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 2D.'
)

job_api.submit(
  analytic_2d_n_sc,
  mu_t,
  chi_ex.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 2D.'
)

job_api.process()

