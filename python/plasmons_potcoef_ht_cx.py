from common import *

N_u0 = 1 << 10
#N_u1 = (1 << 2) + 1
N_u1 = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 300  # K
sys = system_data(m_e, m_h, eps_r, T)

mu_e = 4 / sys.beta
mu_h = sys.get_mu_h_ht(mu_e)
z = 0.2

t0 = time.time()
#potcoef = plasmon_potcoef_ht_cx_mat(N_u0, N_u1, mu_e, mu_h, sys)
potcoef = plasmon_fmat_ht_cx(z, N_u0, N_u1, mu_e, mu_h, sys)
print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

cx_arr = array(potcoef).reshape((N_u1 * N_u0, N_u1 * N_u0))
cx_arr = cx_arr[::-1, :]

colors = matplotlib.colors.hsv_to_rgb(color_map(cx_arr))

"""
plt.figure(200)

for n, (i, j) in enumerate(itertools.product(range(N_u1), repeat=2)):
    plt.subplot(N_u1, N_u1, n + 1)

    plt.imshow(
        colors[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 * (j + 1)],
        extent=(0, 1, 0, 1),
    )
"""

plt.figure("plasmon_potcoef_ht_000")

plt.imshow(
    colors,
    extent=(0, 1, 0, 1),
)

plt.show()
