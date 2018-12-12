from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
sys = system_data(m_e, m_h, eps_r, T)

N_k = 1 << 9
N_x = 1 << 10

x_max = 100

mu_vec = linspace(-12, 4, 9) / sys.beta
x_vec = logspace(log10(x_max / N_x), log10(x_max), N_x)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, mu_vec.size)
]

for c, mu_e in zip(colors, mu_vec):
    mu_h = sys.get_mu_h(mu_e)

    eb = time_func(plasmon_det_zero_ht, N_k, mu_e, mu_h, sys)
    plt.axhline(y=eb, color=c, linestyle='--')
    print('eb: %f eV' % eb)
    """
    rpot = time_func(plasmon_rpot_v, x_vec, mu_e, mu_h, sys)
    plt.plot(
        x_vec,
        rpot,
        '-',
        color=c,
        label=r'$\mu_e$: %3.2f$\times 10^{%1.0f}$' %
        (mu_e * 10**(-int(log10(mu_e))), log10(mu_e)),
    )
    """

    rpot_ht = time_func(plasmon_rpot_ht_v, x_vec, mu_e, mu_h, sys)
    plt.plot(
        x_vec,
        rpot_ht,
        '-',
        color=c,
        label=r'$\mu_e$: %.2e eV' % mu_e,
    )

rpot_lwl = plasmon_rpot_lwl_v(x_vec, sys.sys_ls, sys)
rpot_cou = plasmon_rpot_lwl_v(x_vec, 1e-8, sys)

plt.plot(x_vec, rpot_lwl, 'b:', label='lwl: $\lambda_{s,0}^{-1}$')
plt.plot(x_vec, rpot_cou, 'r:', label='Coulomb')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')

plt.ylim(-0.05, 0.01)
plt.xlim(0, x_max)
plt.legend(loc=0)

plt.title('Electron-Hole potential in real space\n$T$ = %.0f K' % sys.T)
plt.xlabel('$x$ / nm')
plt.ylabel('$V(x)$ / eV')

plt.show()
