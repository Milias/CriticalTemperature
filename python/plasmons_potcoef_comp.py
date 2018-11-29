from common import *
from mpl_toolkits.mplot3d import Axes3D

N_k = 1 << 6
N_w = 1 << 6

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

N_cols, N_rows = 3, 2

x_max, N_x = 1, N_cols * N_rows
x_list = logspace(-5, log10(x_max), N_x)

#potcoef_lwl = array(plasmon_potcoef_lwl_v(kk_prod, sys_ls, sys)).reshape(N_k, N_k)[::-1, :]

for ii, mu_e in enumerate(x_list):
    mu_h = sys.m_eh * mu_e
    q = 2 * sqrt(2 * sys.m_e * mu_e) / sys.c_hbarc
    u_q = 1 / (1 + q)

    u0_vec, u1_vec = linspace(1 / N_k, 1 - 1 / N_k, N_k), linspace(
        -1 + 1 / N_w, 1 - 1 / N_w, N_w)
    w_vec, k_vec = u1_vec / (1 - u1_vec)**2, (1 - u0_vec) / u0_vec
    wkwk_prod = list(itertools.product(w_vec, k_vec, repeat=2))

    t0 = time.time()

    potcoef_static = imag(
        array(plasmon_potcoef_cx_v(wkwk_prod, mu_e, mu_h, sys, 1e-12)).reshape(
            N_k * N_w, N_k * N_w)[::-1, :])

    print('[%e] Elapsed: %.2fs' % (mu_e, time.time() - t0))

    V_q = 0.5  #-2 * plasmon_green(0, q, mu_e, mu_h, 0, sys, 1e-12)[0]
    #potcoef_static_clipped = clip(potcoef_static, 0, 2 * V_q)
    potcoef_static_clipped = potcoef_static
    """
    ax = plt.subplot(N_rows, N_cols, ii + 1, projection='3d')
    K0, K1 = meshgrid(k_vec, k_vec)

    ax.plot_surface(K0, K1, potcoef_static_clipped)
    """
    plt.subplot(N_rows, N_cols, ii + 1)
    """
    plt.axhline(y=q, color='b')
    plt.axvline(x=q, color='b')
    """

    plt.axhline(y=u_q, color='b')
    plt.axvline(x=u_q, color='b')

    plt.imshow(
        potcoef_static_clipped,
        cmap=cm.bwr,
        aspect='auto',
        extent=(1 / N_k, 1 - 1 / N_k, 1 / N_k, 1 - 1 / N_k),
        #extent=(k0 / N_k, k0, k0 / N_k, k0),
        norm=SymLogNorm(linthresh=1e-4),
    )
    if mu_e > 1e-2:
        plt.title('Static: $\mu_e: %.2f$' % mu_e)
    else:
        plt.title('Static: $\mu_e: %.2e$' % mu_e)

#plt.subplot(N_cols, N_rows, 1)
#plt.title('Long-wavelength limit')

#plt.title()
#plt.xlabel()
#plt.ylabel()

#plt.legend(loc=0)

plt.savefig('plots/potcoef_comp.eps')

plt.show()
