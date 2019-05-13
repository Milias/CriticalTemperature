from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
#m_e, m_h, eps_r, T = 0.5, 1, 1, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

N_x = 1 << 10

x_max = 20

mu_vec = logspace(-4, 3, 16)
#x_vec = logspace(log10(x_max / N_x), log10(x_max), N_x)
"""
real_potcoef_lwl = array([
    -sys.sys_ls * sys.c_aEM / sys.eps_r * sys.c_hbarc * pot_limit_2d(x * sys.sys_ls)
    for x in x_vec
])

real_potcoef_cou = -sys.c_aEM / sys.eps_r * sys.c_hbarc / x_vec
"""

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, mu_vec.size)
]

x_val = 2

for c, mu_e in zip(colors, mu_vec):
    mu_h = sys.m_eh * mu_e
    q = 2 * sqrt(2 * sys.m_e * mu_e) / sys.c_hbarc

    t0 = time.time()
    """
    real_potcoef = array(
        plasmon_real_potcoef_k(x_vec, mu_e, mu_h, sys.v_1, sys))
    real_potcoef = -array([(k * plasmon_green(0, k, mu_e, mu_h, 0, sys)[0] -
                            sys.c_aEM / sys.eps_r * 2 * pi * sys.c_hbarc) *
                           scipy.special.jv(0, k * x_val) for k in x_vec])
    real_potcoef = array([(k * plasmon_green(0, k, mu_e, mu_h, 0, sys, 1e-5)[0] +
                            sys.c_aEM / sys.eps_r * sys.c_hbarc)
                           for k in x_vec])
    """
    x_vec = logspace(log10(q * (1 + 1e-10)), log10(x_max), N_x)
    real_potcoef = array(
        [(k * plasmon_green(0, k, mu_e, mu_h, 0, sys, 1e-12)[0] +
          (sys.c_aEM / sys.eps_r * sys.c_hbarc /
           (1 + sys.c_aEM / sys.eps_r * 0.5 / pi *
            (sys.m_e + sys.m_h) / sys.c_hbarc / k) if k < q else 0) +
          (sys.c_aEM / sys.eps_r * sys.c_hbarc if k > q else 0))
         for k in x_vec])

    k_kink = 2 * sqrt(2 * sys.m_e * mu_e) / sys.c_hbarc
    """
    plt.axhline(
        y=-sys.c_aEM / sys.eps_r * 2 * pi * sys.c_hbarc,
        linestyle='--',
        color=c)

    plt.plot(
        x_vec,
        -sys.c_aEM / sys.eps_r * 2 * pi * sys.c_hbarc /
        (1 + sys.c_aEM / sys.eps_r * 0.5 / pi *
         (sys.m_e + sys.m_h) / sys.c_hbarc / x_vec),
        linestyle=':',
        color=c)
    """

    plt.axvline(x=k_kink, color=c)

    print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

    plt.plot(
        x_vec,
        real_potcoef,
        '-',
        color=c,
        label=r'$\mu_e$: %3.2f$\times 10^{%1.0f}$' %
        (mu_e * 10**(-int(log10(mu_e))), log10(mu_e)))

#plt.plot(x_vec, real_potcoef_lwl, 'b--', label='LWL: $\lambda_{s,0}^{-1}$')
#plt.plot(x_vec, real_potcoef_cou, 'r:', label='Coulomb')

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
#plt.axvline(x=k_kink, color='b')
"""
for j0 in scipy.special.jn_zeros(0, ceil(x_val * x_max / pi)):
    plt.axvline(x=j0 / x_val, color='b', linestyle=':')
"""

#plt.ylim(-0.01, 0.002)
plt.xlim(0, x_max)
plt.legend(loc=0)

plt.title('Electron-Hole potential in real space')
plt.xlabel('$x$ / nm')
plt.ylabel('$V(x)$ / eV')

plt.show()
