from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N = 1<<12
N2 = N*N
bs = N2 // 64

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1 # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print((sys.m_e, sys.m_h))
mu_e, v_1 = 1, 1e-6
mu_h = sys.m_eh * mu_e

if v_1 > 0:
  kmax = plasmon_kmax(mu_e, mu_h, v_1, sys)
else:
  kmax = 1

wmax = plasmon_wmax(kmax, mu_e, sys)

factor = 1
w0, k0 = factor * wmax, factor * kmax
print((wmax, kmax))
print((w0, k0))

w, k = linspace(-w0, w0, N), linspace(-k0, k0, N)
W, K = map(operator.itemgetter(0), itertools.product(w, k)), map(operator.itemgetter(1), itertools.product(w, k))

batch = job_api.new_batch(name, description)

green = job_api.submit(
  plasmon_green,
  W,
  K,
  itertools.repeat(mu_e, N2),
  itertools.repeat(mu_h, N2),
  itertools.repeat(v_1, N2),
  itertools.repeat(sys, N2),
  itertools.repeat(5e-2, N2),
  size = N2,
  block_size = bs,
  desc = 'Plasmon Green function.'
)

job_api.process()

