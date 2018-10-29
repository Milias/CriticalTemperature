from common import *
from plasmon_utils import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N_k = 1 << 11

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

name = 'Plasmon Green Function'
description = """Plasmon Green function.

Parameters: $m_e$ = $%f$, $m_h$ = $%f$, $\eps_r$ = $%f$, $T$ = $%f$ K""" % (
    sys.m_e, sys.m_h, sys.eps_r, sys.T)

v_1, ls = 0.1, 1e-3

z_wf = wf_2d_E_cou_py(sys) * sys.energy_th

print('z_wf: %f' % z_wf)

z_arr = logspace(-4, 2, 1 << 7)
x_list = logspace(-2, -1, 8)
y_list = logspace(-1, 2, 8)
z_list = logspace(-1, 1, 8)

for v_1 in x_list:
    k_pl_max = max(1, ls)
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
    z_adj_arr = array(z_arr, copy=True) / energy_fermi

    mat_det_list = plasmon_sysmat_det_lwl_v(z_adj_arr, list(kk_prod),
                                            list(ids_prod), dk, N_k, ls, v_1,
                                            sys)

    print('[%e,%e], Elapsed: %.2fs' % (ls, v_1, time.time() - t0))

    plt.semilogx(
        real(z_arr),
        real(mat_det_list),
        '.-',
        label=r'$\lambda_s^{-1}$: %.2e, $V_1^{-1}$: %.2e' % (ls, v_1))

plt.axhline(y=0, color='k')
plt.legend(loc=0)
plt.show()
