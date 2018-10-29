from common import *
from plasmon_utils import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N_k = 1 << 13
N_w = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (
    sys.m_e, sys.m_h, sys.eps_r, sys.T)

mu_e, v_1, ls = 1, 1 / 50, 1e-3
mu_h = sys.m_eh * mu_e

z_wf = wf_2d_E_cou_py(sys) * sys.energy_th

print('z_wf: %f' % z_wf)

k_pl_max = max(1, ls)
kmax = k_pl_max
print('plasmon_kmax: %f' % k_pl_max)
k0 = 1 * kmax
k1 = k0

print('kmax: %f, k0: %f' % (kmax, k0))

k = linspace(0.0, k0, N_k)
kk_prod = list(itertools.product(k, repeat=2))

r_k = range(N_k)
ids_prod = list(itertools.product(r_k, repeat=2))

dk = k[1] - k[0]

energy_fermi = 8 * pi**2 * (sys.c_aEM * v_1)**2 * sys.m_p
print('E_F: %4.3f eV' % energy_fermi)

z = plasmon_sysmat_det_zero_lwl(kk_prod, ids_prod, dk, N_k, ls, v_1, sys)

#z = - 5

print('eB: %f eV' % (z * energy_fermi))

if isnan(z):
    print('eB is NaN')
    exit()

t0 = time.time()
result = plasmon_sysmat_lwl_m(z, kk_prod, ids_prod, dk, ls, v_1, sys)

print('[%e, %e] (%f), Elapsed: %.2fs' % (ls, v_1, z, time.time() - t0))

p = 1e-1
potcoef_arr = array(result).reshape(N_k, N_k)[::-1,:]
potcoef_arr = clip(potcoef_arr, amin(potcoef_arr), p)

plt.imshow(
    potcoef_arr,
    cmap=cm.hot,
    aspect='auto',
    extent=(amin(k), amax(k), amin(k), amax(k)),
    norm = SymLogNorm(linthresh = 1e-4)
    #norm=LogNorm()
)

plt.axis([amin(k), amax(k), amin(k), amax(k)])

plt.colorbar()

plt.show()
