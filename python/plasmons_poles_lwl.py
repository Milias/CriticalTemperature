from common import *
from plasmon_utils import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N_k = 1 << 12
N_w = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (
    sys.m_e, sys.m_h, sys.eps_r, sys.T)

print((sys.m_e, sys.m_h))
mu_e, v_1 = 1, 0.05
mu_h = sys.m_eh * mu_e

z_wf = wf_2d_E_cou_py(sys) * sys.energy_th

print('z_wf: %f' % z_wf)

x_list = logspace(-2, 1, 12)

plt.axhline(y=z_wf, color='g')

z_list = []
for ls in x_list:
    k_pl_max = max(0.5, ls)
    kmax = k_pl_max
    print('plasmon_kmax: %f' % k_pl_max)
    k0 = 6 * kmax
    k1 = k0

    print('kmax: %f, k0: %f' % (kmax, k0))

    k = linspace(0.0, k0, N_k)
    kk_prod = itertools.product(k, repeat=2)

    r_k = range(N_k)
    ids_prod = itertools.product(r_k, repeat=2)

    dk = k[1] - k[0]

    energy_fermi = 8 * pi**2 * (sys.c_aEM * v_1)**2 * sys.m_p
    print('E_F: %4.3f eV' % energy_fermi)

    t0 = time.time()
    z = plasmon_sysmat_det_zero_lwl(
        list(kk_prod), list(ids_prod), dk, N_k, ls, v_1, sys)

    z *= energy_fermi

    print('[%e] (%f), Elapsed: %.2fs' % (ls, z, time.time() - t0))

    z_list.append(z)

plt.axhline(y=0, color='k')
z_arr = array(z_list)
plt.plot(x_list, real(z_arr), 'r.-')

plt.show()
