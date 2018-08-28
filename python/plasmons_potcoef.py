from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<5
N_k = 1<<3
N_w = 1<<3

N_total = N_k * N_k * N_w

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1 # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print((sys.m_e, sys.m_h))
mu_e, mu_h = 2, 1

w0, k0 = 2e-2, 1e-3

w, k = linspace(-w0, w0, N_w), linspace(-k0, k0, N_k)
W, K1, K2 = meshgrid(w, k, k, indexing = 'ij')

batch = job_api.new_batch(name, description)

green = job_api.submit(
  plasmon_potcoef,
  W.flatten(),
  K1.flatten(),
  K2.flatten(),
  itertools.repeat(mu_e, N_total),
  itertools.repeat(mu_h, N_total),
  itertools.repeat(sys, N_total),
  size = N_total,
  block_size = bs,
  desc = 'Plasmon: Potential matrix elements.'
)

job_api.process()

