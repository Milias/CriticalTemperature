from common import *

job_api = JobAPI()

N = 1<<12
bs = 1<<2

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300 # K

sys = system_data(m_e, m_h, eps_r, T)

print('%f nm, %f meV' % (sys.lambda_th * 1e9, sys.energy_th * 1e3))

n = 3e23 * sys.lambda_th**3
a = 5

#exit()

x = logspace(log10(1e23), log10(7e25), N) * sys.lambda_th**3
#x = linspace(0, 10, N)

job_api.submit(
  analytic_mu,
  N,
  x,
  itertools.repeat(sys, N),
  bs = bs,
  desc = 'Testing job'
)

job_api.process()

