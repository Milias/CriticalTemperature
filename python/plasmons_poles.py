from common import *
from plasmon_utils import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N_k = 1 << 8
N_w = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

z_wf = wf_2d_E_cou_py(sys) * sys.energy_th

print('z_wf: %f' % z_wf)

x_list = logspace(-3, 1, 32)

plt.axhline(y=z_wf, color='g')

mu_e, v_1 = 1, 0.05

x_list = logspace(-2, 2, 16)

z_list = []
z_lwl_list = []
for mu_e in x_list:
    mu_h = sys.m_eh * mu_e

    k_pl_max = plasmon_kmax(mu_e, mu_h, v_1, sys)
    kmax = 1 #max(1, k_pl_max)
    print('k_pl_max: %f' % k_pl_max)

    w0, k0 = 0, 28 * kmax
    k1 = k0 * (1 - 1e-5)

    print('kmax: %f, k0: %f, N_k: %d' % (kmax, k0, N_k))

    w, k = linspace(-w0, w0, N_w), linspace(k0 / N_k, k0, N_k)
    wkwk_prod = itertools.product(w, k, repeat=2)

    r_w, r_k = range(N_w), range(N_k)
    ids_prod = itertools.product(r_w, r_k, repeat=2)

    dw, dk = w[1] - w[0] if N_w > 1 else 1.0, k[1] - k[0]

    t0 = time.time()
    z = plasmon_sysmat_det_zero(
        list(wkwk_prod), list(ids_prod), dk, dw, N_k, N_w, mu_e, mu_h, v_1,
        sys)

    z_lwl = plasmon_sysmat_det_zero_lwl(
        list(kk_prod), list(ids_prod), dk, N_k, ls, v_1, sys)

    energy_fermi = 8 * pi**2 * (sys.c_aEM * v_1)**2 * sys.m_p
    print('E_F: %4.3f eV' % energy_fermi)

    z *= energy_fermi
    z_lwl *= energy_fermi
    print('[%e,%e] z: %f+%fj, Elapsed: %.2fs' % (mu_e, v_1, real(z), imag(z),
                                                 time.time() - t0))

    z_list.append(z)
    z_lwl_list.append(z_lwl)

z_arr = array(z_list)

plt.axhline(y=0, color='k')
plt.plot(x_list, real(z_arr), 'r.-', label=r'Full, $V_1$: %.2f' % (v_1, ))
plt.plot(x_list, real(z_lwl_arr), 'b.-', label=r'LWL, $V_1$: %.2f' % (v_1, ))

#plt.plot(x_list, imag(z_arr), 'b.-')
"""
plt.loglog(x_list, -real(z_arr), 'r.-')
#plt.semilogx(x_list, imag(z_arr), 'b.-')
"""

plt.show()
