from common import *

N_u0 = 1 << 11
N_u1 = (1 << 11)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

u_max = 1

u0, du0 = linspace(u_max / N_u0, u_max - u_max / N_u0, N_u0, retstep=True)

if N_u1 > 1:
    u1, du1 = linspace(
        -u_max + u_max / N_u1, u_max - u_max / N_u1, N_u1, retstep=True)
else:
    u1, du1 = array([0.0]), float('nan')

wk_vec = list(itertools.product(u1 / (1 - u1**2), 1.0 / (1 - u0) - 1.0))

#plt.axvline(x=z_sys_lwl, color='m')
#plt.axvline(x=z_cou_lwl, color='g')
plt.axhline(y=0, color='w', linestyle='--')

mu_e = 1e1
mu_h = sys.m_eh * mu_e

t0 = time.time()
green = plasmon_green_v(wk_vec, mu_e, mu_h, sys, 1e-5)

print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

green_arr = array(green).reshape((N_u1, N_u0)).T
u1_arr, u0_arr = meshgrid(u1, u0)

green_r, green_ph = abs(green_arr), angle(green_arr)

green_h = 0.5 + 0.5 * green_ph / pi
green_s = 0.9 * ones_like(green_r)
green_v = green_r / (1.0 + green_r)

green_hsv = array([green_h, green_s, green_v]).T

green_colors = matplotlib.colors.hsv_to_rgb(green_hsv)

plt.imshow(
    green_colors,
    aspect='auto',
    extent=(0, u_max, -u_max, u_max),
)

plt.imshow(
    green_colors[:, ::-1, :],
    aspect='auto',
    extent=(-u_max, 0, -u_max, u_max),
)

plt.contour(u1_arr, u0_arr, green_r, 0, cmap=cm.cool)
plt.contour(-u1_arr, u0_arr, green_r[::-1, :], 0, cmap=cm.cool)

plt.savefig('plots/green_cx_2d.eps')

plt.show()
