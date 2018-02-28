from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N = 1<<13
bs = 1<<2

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

print('%f nm, %f meV' % (sys.lambda_th * 1e9, sys.energy_th * 1e3))

n = 3e23 * sys.lambda_th**3
a = -0.001

#batch = job_api.load_batch()
batch = job_api.new_batch('Testing batch', 'Testing hashing')
#print(batch.api_data)

#exit()

x0, x1 = log10(1e23 * sys.lambda_th**3), log10(1e25 * sys.lambda_th**3)
x = iter_linspace(x0, x1, N, func = iter_log_func)
x2 = iter_linspace(-100, 200, N)

job_api.submit(
  analytic_mu,
  x,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = r"""Computing chemical potential neglecting fluctuations.

  Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" %
  (sys.m_e, sys.m_h, sys.eps_r, sys.T)
)

init_mu = job_api.submit(
  analytic_mu_init_mu,
  x,
  itertools.repeat(a, N),
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = r"""Computing chemical potential neglecting fluctuations.

  Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" %
  (sys.m_e, sys.m_h, sys.eps_r, sys.T)
)

job_api.process()

job_api.submit(
  analytic_mu_f,
  init_mu.result,
  itertools.repeat(a, N),
  x,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = r"""Computing chemical potential neglecting fluctuations.

  Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" %
  (sys.m_e, sys.m_h, sys.eps_r, sys.T)
)

job_api.submit(
  ideal_mu,
  x,
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = r"""Computing chemical potential neglecting fluctuations.

  Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" %
  (sys.m_e, sys.m_h, sys.eps_r, sys.T)
)

job_api.process()

