from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N = 1<<17
bs = 1<<8

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

print('%f nm, %f meV' % (sys.lambda_th * 1e9, sys.energy_th * 1e3))

n = 3e23 * sys.lambda_th**3
a = 5

#batch = job_api.resume_last_batch()
batch = job_api.new_batch('Testing batch', 'Testing hashing')
#print(batch.api_data)

#exit()

x0, x1 = log10(1e23), log10(7e25)
x = iter_linspace(x0, x1, N, func = iter_log_func)

ideal_ls = job_api.submit(
  ideal_ls,
  x,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  description = r"""Computing lambda_s for analytic_a_ls.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K."""
)

job_api.process()

job_api.submit(
  analytic_a_ls,
  ideal_ls.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = r"""Computing scattering length.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K.""")

job_api.process()

