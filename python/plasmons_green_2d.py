from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<11
N = 1<<11
N2 = N*N

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1 # K
#m_e, m_h, eps_r, T = 0.28, 0.28, 6.56, 1 # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print((sys.m_pe, sys.m_ph, 1 + sys.m_eh, 1 + 1/sys.m_eh))

mu_e, v_1 = 1, 1e2
mu_h = sys.m_eh * mu_e

print((mu_e, mu_h))

mu_t = mu_e + mu_h

w0, k0 = 4.0, 4.0

k_vec = linspace(0, k0, N)
t1 = time.time()
w_vec = array([plasmon_disp(k, mu_e, mu_h, v_1, sys) for k in k_vec])
print((time.time() - t1)/N)

plt.plot(k_vec, w_vec, 'r-')
plt.show()

exit()
w, k = linspace(-w0, w0, N), linspace(-k0, k0, N)
W, K = meshgrid(w, k, indexing = 'ij')


batch = job_api.new_batch(name, description)

green = job_api.submit(
  plasmon_green,
  W.flatten(),
  K.flatten(),
  itertools.repeat(mu_e, N2),
  itertools.repeat(mu_h, N2),
  itertools.repeat(v_1, N2),
  itertools.repeat(sys, N2),
  itertools.repeat(2e-5, N2),
  size = N2,
  block_size = bs,
  desc = 'Plasmon Green function.'
)

job_api.process()

