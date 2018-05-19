from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<0
N = 1<<12

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 10 # K
sys = system_data(m_e, m_h, eps_r, T)

print((sys.m_pe, sys.m_ph))

chi_ex = -100
mu_e = ideal_2d_mu_v(1.1 * chi_ex, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)
mu_t = mu_e + mu_h

#print((mu_e, mu_h, mu_t))

#print(fluct_2d_I2(1871.52 - mu_e - mu_h - chi_ex * exp(-19.196536), sys.m_sigma * 1871.52, mu_e, mu_h, sys))
#print(fluct_2d_I2(1 - mu_e - mu_h - chi_ex * 10, sys.m_sigma * 1, mu_e, mu_h, sys))
#print(fluct_2d_I2_p(10, 1, chi_ex, mu_e, mu_h, sys))
#print(fluct_2d_n_ex(chi_ex, mu_e, mu_h, sys))

#t0 = time.time()
#print(fluct_2d_n_sc_v2(chi_ex, mu_e, mu_h, sys))
#print('v2: %f s' % (time.time() - t0))

#t0 = time.time()
#print(fluct_2d_n_sc(chi_ex, mu_e, mu_h, sys))
#print('v1: %f s' % (time.time() - t0))

n = 50
print('n: %.1e cm^-2' % (1e-4 * n * sys.lambda_th**-2))

t0 = time.time()

mu_e, ls = fluct_2d_mu(n, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)
chi_ex = wf_2d_E_lim_py(ls, sys)

print('mu_e: %f, mu_h: %f, ls: %e, chi_ex: %f' % (mu_e, mu_h, ls, chi_ex))

n_id = ideal_2d_n(mu_e, sys.m_pe)
n_ex = fluct_2d_n_ex(chi_ex, mu_e, mu_h, sys)
n_sc = fluct_2d_n_sc(chi_ex, mu_e, mu_h, sys)

print('n_id: %e, n_ex: %e, n_sc: %e' % (n_id, n_ex, n_sc))

print('%.2f s' % (time.time() - t0))

exit()

#"""
z = 1
t = 1
E = 0

x1 = E - mu_t + chi_ex * linspace(1, global_eps, N)
x2 = linspace(global_eps, 1, N)
x = hstack((x1, E - mu_t - chi_ex * x2))

#y = array([fluct_2d_I2_b(x - mu_t - chi_ex * z, sys.m_sigma * x, mu_e, mu_h, sys.m_pe) for x in x]).reshape(N, 2)
#y = array([fluct_2d_I2_b(x - mu_t - chi_ex * exp(-t), sys.m_sigma * x, mu_e, mu_h, sys.m_pe) for x in x]).reshape(N, 2)
#y = y[:, 0] + 1j * y[:, 1]
#y2 = y[:, 1] - y[:, 0]

#yt = array([fluct_2d_I2(x - mu_t - chi_ex * exp(-t), sys.m_sigma * x, mu_e, mu_h, sys) for x in x])

y1 = array([fluct_2d_I2(x, sys.m_sigma * E, mu_e, mu_h, sys) for x in x1])
y2 = array([fluct_2d_I2_p(x, E, chi_ex, mu_e, mu_h, sys) for x in x2])
y = hstack((y1, y2))

#t0 = time.time()
#yz_v2 = array([fluct_2d_I2_p(x, z, chi_ex, mu_e, mu_h, sys) for x in x])
#print(time.time() - t0)

#t0 = time.time()
#yz = array([fluct_2d_I2(z - mu_t - chi_ex * x, sys.m_sigma * z, mu_e, mu_h, sys) for x in x])
#print(time.time() - t0)

#exit()

#print(y)

#x = linspace(0, 1, N) * sys.get_z1(E, mu_t)
#y = array([1 / (1 + 2 * (sys.get_z1(E, mu_t) - x) * fluct_2d_I2_dz(x, E, mu_e, mu_h, sys)) for x in x])

#plt.plot(x, y2, 'g.-')
plt.plot(x, real(y), 'r-')
plt.plot(x, imag(y), 'b-')
#plt.plot(x, real(yt), 'r.-')
#plt.plot(x, imag(yt), 'b.-')
#plt.plot(x, real(yz), 'r-')
#plt.plot(x, imag(yz), 'b-')
#plt.plot(x, real(yz_v2), 'm--')
#plt.plot(x, imag(yz_v2), 'g--')
plt.show()
exit()

x = linspace(mu_t, 0, N)
y = array([complex(fluct_2d_n_ex(z, mu_e, mu_h, sys), 0) for z in x])

plt.plot(x, real(y), 'r-')
plt.plot(x, imag(y), 'b-')
plt.show()

exit()
#"""

name = 'Testing 2D'
description = """Testing 2D analytic densities.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

print('mu_e: %f, mu_h: %f' % (mu_e, mu_h))

mu_t = iter_linspace(10 * chi_ex, chi_ex - global_eps, N)

batch = job_api.new_batch(name, description)

mu_e = job_api.submit(
  ideal_2d_mu_v,
  mu_t,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Electronic chemical potential.'
)

job_api.process()

mu_h = job_api.submit(
  ideal_2d_mu_h,
  mu_e.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Hole chemical potential.'
)

job_api.process()

n_id = job_api.submit(
  ideal_2d_n,
  mu_e.result,
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 2D.'
)

n_ex = job_api.submit(
  fluct_2d_n_ex,
  itertools.repeat(chi_ex, N),
  mu_e.result,
  mu_h.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 2D.'
)

n_sc = job_api.submit(
  fluct_2d_n_sc,
  itertools.repeat(chi_ex, N),
  mu_e.result,
  mu_h.result,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 2D.'
)

job_api.process()

