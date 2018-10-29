from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

v_1 = 0.05
N_x = 1 << 10

x_max = 10

mu_vec = logspace(-4, 2, 8)
x_vec = linspace(x_max / N_x, x_max, N_x)

for mu_e in mu_vec:
    mu_h = sys.m_eh * mu_e

    k_pl_max = plasmon_kmax(mu_e, mu_h, v_1, sys)
    print('plasmon_kmax: %f' % k_pl_max)

    energy_fermi = 8 * pi**2 * (sys.c_aEM * v_1)**2 * sys.m_p
    print('E_F: %4.3f eV' % energy_fermi)

    t0 = time.time()

    real_potcoef = array(plasmon_real_potcoef_k(x_vec, mu_e, mu_h, v_1,
                                                sys)) * energy_fermi

    print('[%e,%e], Elapsed: %.2fs' % (mu_e, v_1, time.time() - t0))

    plt.plot(
        x_vec,
        real_potcoef,
        '-',
        label=r'$\mu_e$: %.2e, $V_1^{-1}$: %.2e' % (mu_e, v_1))

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='m')

plt.ylim(-10, None)
plt.legend(loc=0)
plt.show()
