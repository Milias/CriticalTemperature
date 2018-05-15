from common import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

bs = 1<<4
N = 1<<12

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 30 # K
sys = system_data(m_e, m_h, eps_r, T)

print((sys.m_pe, sys.m_ph))

chi_ex = -1
mu_e = ideal_2d_mu_v(-2, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)
mu_t = mu_e + mu_h

print((mu_e, mu_h, mu_t))

#print(fluct_2d_I2(1871.52 - mu_e - mu_h - chi_ex * exp(-19.196536), sys.m_sigma * 1871.52, mu_e, mu_h, sys))
#print(fluct_2d_I2(1 - mu_e - mu_h - chi_ex * 10, sys.m_sigma * 1, mu_e, mu_h, sys))
#print(fluct_2d_I2_p(10, 1, chi_ex, mu_e, mu_h, sys))
#print(fluct_2d_n_sc(chi_ex, mu_e, mu_h, sys))
#exit()

z = 1
t = 1

x = linspace(1, 200, N)

#y = array([fluct_2d_I2_b(x - mu_t - chi_ex * z, sys.m_sigma * x, mu_e, mu_h, sys.m_pe) for x in x]).reshape(N, 2)
#y = array([fluct_2d_I2_b(x - mu_t - chi_ex * exp(-t), sys.m_sigma * x, mu_e, mu_h, sys.m_pe) for x in x]).reshape(N, 2)
#y = y[:, 0] + 1j * y[:, 1]
#y2 = y[:, 1] - y[:, 0]

#yt = array([fluct_2d_I2(x - mu_t - chi_ex * exp(-t), sys.m_sigma * x, mu_e, mu_h, sys) for x in x])

t0 = time.time()
yz = array([fluct_2d_I2(z - mu_t - chi_ex * x, sys.m_sigma * z, mu_e, mu_h, sys) for x in x])
print(time.time() - t0)

t0 = time.time()
yz_v2 = array([fluct_2d_I2_p(x, z, chi_ex, mu_e, mu_h, sys) for x in x])
print(time.time() - t0)

#print(y)

#plt.plot(x, y2, 'g.-')
#plt.plot(x, real(yt), 'r.-')
#plt.plot(x, imag(yt), 'b.-')
plt.plot(x, real(yz), 'r-')
plt.plot(x, imag(yz), 'b-')
plt.plot(x, real(yz_v2), 'm--')
plt.plot(x, imag(yz_v2), 'g--')
plt.show()
exit()

x = linspace(mu_t, 0, N)
y = array([complex(fluct_2d_n_ex(z, mu_e, mu_h, sys), 0) for z in x])

plt.plot(x, real(y), 'r-')
plt.plot(x, imag(y), 'b-')
plt.show()

exit()

name = 'Testing 2D'
description = """Testing 2D analytic densities.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (sys.m_e, sys.m_h, sys.eps_r, sys.T)

mu_t = -1
mu_e = ideal_2d_mu_v(mu_t, sys)
mu_h = ideal_2d_mu_h(mu_e, sys)

print('mu_e: %f, mu_h: %f' % (mu_e, mu_h))

chi_ex = iter_linspace(mu_t + global_eps, -global_eps, N)

batch = job_api.new_batch(name, description)

n_id = job_api.submit(
  ideal_2d_n,
  itertools.repeat(mu_e, N),
  itertools.repeat(sys.m_pe, N),
  size = N,
  block_size = bs,
  desc = 'Ideal density in 2D.'
)

n_ex = job_api.submit(
  analytic_2d_n_ex,
  itertools.repeat(mu_t, N),
  chi_ex,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Excitonic density in 2D.'
)

n_sc = job_api.submit(
  analytic_2d_n_sc,
  itertools.repeat(mu_t, N),
  chi_ex,
  itertools.repeat(sys, N),
  size = N,
  block_size = bs,
  desc = 'Scattering density in 2D.'
)

job_api.process()

