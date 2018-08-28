from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<0
N = 1<<10

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 3 # K
sys = system_data(m_e, m_h, eps_r, T)

chi_ex = wf_2d_E_lim_py(ideal_2d_ls(1e20, sys), sys)
print(chi_ex)
print(analytic_2d_n_sc(chi_ex - 1e-1, chi_ex, sys))
exit()

name = 'Testing 2D'
description = """Testing 2D analytic densities.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

#n0, n1 = 1e12 * sys.lambda_th**2, 1e16 * sys.lambda_th**2
n0, n1 = 1e-3, 10
n = iter_linspace(log10(n0), log10(n1), N, func = iter_log_func)

batch = job_api.new_batch(name, description)

mu_ls = job_api.submit(
  analytic_2d_mu,
  n,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Chemical potential (analytic) in 2D.'
)

job_api.process()

mu_ls_arr = array(mu_ls.result)

mu_h = job_api.submit(
  ideal_2d_mu_h,
  mu_ls_arr[:, 0],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Chemical potential for holes in 2D.'
)

chi_ex = job_api.submit(
  wf_2d_E_lim_py,
  mu_ls_arr[:, 1],
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Binding energy in 2D.'
)

job_api.process()

mu_h_arr = array(mu_h.result)
mu_t = mu_ls_arr[:, 0] + mu_h_arr

n_id = job_api.submit(
  ideal_2d_n,
  mu_ls_arr[:, 0],
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 2D.'
)

n_ex = job_api.submit(
  analytic_2d_n_ex,
  mu_t,
  chi_ex.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 2D.'
)

n_sc = job_api.submit(
  analytic_2d_n_sc,
  mu_t,
  chi_ex.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 2D.'
)

job_api.process()

