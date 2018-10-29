from common import *
from plasmon_utils import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N_k = 1<<10
N_w = (1<<0)

N_total = N_w * N_k * (N_k + 1) // 2
bs = N_total // 256

print('Total: %d, BlockSize: %d, Batches: %d' % (N_total, bs, N_total // bs))

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1 # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print((sys.m_e, sys.m_h))
mu_e, v_1 = 1, 1e2
mu_h = sys.m_eh * mu_e

kmax = plasmon_kmax(mu_e, mu_h, v_1, sys)

print(kmax)

w0, k0 = 0, 32 * kmax
k1 = k0 * (1 - 1e-3)

w, k = linspace(-w0, w0, N_w), linspace(k0 / N_k, k0, N_k)
wkk_iter = PlasmonPotcoefIterator(w, k)

N_z = 1<<5
N_total_2 = N_w**2 * N_k**2 * N_z
bs = N_total_2 // (1<<10)

print((N_total, N_total_2, bs))

r_w, r_k = range(N_w), range(N_k)

ids_prod = itertools.product(r_w, r_k, r_w, r_k)
ids_list = list(ids_prod)

wkwk_prod = itertools.product(w, k, w, k)
wkwk_list = list(wkwk_prod)

z_list = -logspace(1, 3, N_z)

map_list = [
  map(operator.itemgetter(1), itertools.product(z_list, ids_list)),
  map(operator.itemgetter(1), itertools.product(z_list, wkwk_list)),
  map(operator.itemgetter(0), itertools.product(z_list, wkwk_list)),
]

dw, dk = w[1] - w[0] if N_w > 1 else 1.0, k[1] - k[0]

exit()

batch = job_api.new_batch(name, description)

job_api.submit(
  plasmon_sysmatelem_full,
  map_list[0],
  map_list[1],
  itertools.repeat(dw),
  itertools.repeat(dk),
  map_list[2],
  itertools.repeat(mu_e),
  itertools.repeat(mu_h),
  itertools.repeat(v_1),
  itertools.repeat(sys),
  size = N_total_2,
  block_size = bs,
  desc = 'Plasmon: System matrix elements.'
)

job_api.process()

