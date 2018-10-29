from common import *
from plasmon_utils import *

api_token = 'F4RfBdBNx1fLqH2jTsDoJP9xqERAe5z/ummsn16tDdKRmeOtQTZq/htBvJou5FCOF5EaYZw6U4xEv7/EHa2f+w=='
job_api = JobAPI(api_token)

N_k = 1 << 11
N_w = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

z_wf = wf_2d_E_cou_py(sys) * sys.energy_th

print('z_wf: %f' % z_wf)

mu_e, v_1 = 1, 0.05

z_arr = logspace(-2, 2, 1 << 7) + 0j
x_list = logspace(-3, 2, 8)

for mu_e in x_list:
    mu_h = sys.m_eh * mu_e

    k_pl_max = plasmon_kmax(mu_e, mu_h, v_1, sys)
    kmax = max(1, k_pl_max)
    print('plasmon_kmax: %f' % k_pl_max)
    w0, k0 = 0, 12 * kmax
    k1 = k0 * (1 - 1e-5)

    print('kmax: %f, k0: %f' % (kmax, k0))

    w, k = linspace(-w0, w0, N_w), linspace(k0 / N_k, k0, N_k)
    wkwk_prod = itertools.product(w, k, repeat=2)

    r_w, r_k = range(N_w), range(N_k)
    ids_prod = itertools.product(r_w, r_k, repeat=2)

    dw, dk = w[1] - w[0] if N_w > 1 else 1.0, k[1] - k[0]

    energy_fermi = 8 * pi**2 * (sys.c_aEM * v_1)**2 * sys.m_p
    print('E_F: %4.3f eV' % energy_fermi)

    z_adj_arr = array(z_arr, copy=True) / energy_fermi

    t0 = time.time()
    mat_det_list = plasmon_sysmat_det_v(z_adj_arr, list(wkwk_prod),
                                        list(ids_prod), dk, dw, N_k, N_w, mu_e,
                                        mu_h, v_1, sys)

    print('[%e,%e], Elapsed: %.2fs' % (mu_e, v_1, time.time() - t0))

    plt.semilogx(
        real(z_arr),
        real(mat_det_list),
        '.-',
        label=r'$V_1$: %.2f, $\mu_e$: %.2f' % (v_1, mu_e))
    #plt.plot(real(z_arr), imag(mat_det_list), '.--')

plt.axhline(y=0, color='k')
plt.legend(loc=0)
plt.show()
