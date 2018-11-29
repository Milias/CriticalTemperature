from common import *

N_u0_lwl = 1 << 8

N_u0 = 1 << 6
N_u1 = (1 << 2) + 1
#N_u1 = (1 << 0)

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

u_max = 1

u0_lwl, du0_lwl = linspace(
    u_max / N_u0_lwl, 1 - u_max / N_u0_lwl, N_u0_lwl, retstep=True)

u0, du0 = linspace(u_max / N_u0, 1 - u_max / N_u0, N_u0, retstep=True)
u1_nw, du1_nw = array([0.0]), float('nan')

if N_u1 > 1:
    u1, du1 = linspace(-1 + u_max / N_u1, 1 - u_max / N_u1, N_u1, retstep=True)
else:
    u1, du1 = array([0.0]), float('nan')

wk_vec = list(itertools.product(u1 / (1 - u1**2), 1.0 / (1 - u0) - 1.0))

vu_vec = list(itertools.product(u0_lwl, repeat=2))
vuvu_vec = list(itertools.product(u0, u1, repeat=2))
vuvu_nw_vec = list(itertools.product(u0, u1_nw, repeat=2))

r_u0, r_u1 = list(range(N_u0)), list(range(N_u1))
r_u0_lwl = list(range(N_u0_lwl))

id_lwl_vec = list(itertools.product(r_u0_lwl, repeat=2))
id_vec = list(itertools.product(r_u0, r_u1, repeat=2))
id_nw_vec = list(itertools.product(r_u0, itertools.repeat(0, 1), repeat=2))

wkwk_vec = list(itertools.product(u1 / (1 - u1**2), (1 - u0) / u0, repeat=2))

z_cou_lwl = plasmon_det_zero_lwl(vu_vec, id_lwl_vec, N_u0_lwl, du0_lwl, 1e-8,
                                 sys)
z_sys_lwl = plasmon_det_zero_lwl(vu_vec, id_lwl_vec, N_u0_lwl, du0_lwl,
                                 sys.sys_ls, sys)

#plt.axvline(x = z_sys_lwl, color = 'm')
#plt.axvline(x = z_cou_lwl, color = 'g')
#plt.axhline(y = 0, color = 'y')

print('sys_ls: \t%8.6f nm' % (1 / sys.sys_ls))
print('###   lwl    ###')
print('z_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV' % (z_cou_lwl, z_sys_lwl))

mu_e = 1e1
mu_h = sys.m_eh * mu_e

t0 = time.time()
potcoef = plasmon_potcoef_cx_v(wkwk_vec, mu_e, mu_h, sys)
print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

cx_arr_copy = array(potcoef).reshape((N_u0 * N_u1, N_u0 * N_u1))
#cx_arr = zeros_like(cx_arr_copy)
cx_arr = array(cx_arr_copy, copy=True)
"""
for n, (i, j) in enumerate(itertools.product(range(N_u1), repeat=2)):
    if i == j and i < (N_u1 - 1):
        print('%d, %d: 1' % (i, j))
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] -= cx_arr[N_u0 * (i + 1):N_u0 * (i + 2), N_u0 *
                                  (j + 1):N_u0 * (j + 2)]

    elif j > i:
        print('%d, %d: 2' % (i, j))
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 * (j + 1)] -= conj(
            cx_arr[N_u0 * j:N_u0 * (j + 1), N_u0 * i:N_u0 * (i + 1)])

    elif j == 0 and i < (N_u1 - 1):
        print('%d, %d: 3' % (i, j))
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] -= cx_arr[N_u0 * (i + 1):N_u0 * (i + 2), N_u0 *
                                  (j):N_u0 * (j + 1)]

    elif i == (N_u1 - 1) and j < (N_u1 - 1):
        print('%d, %d: 4' % (i, j))
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] -= cx_arr[N_u0 * (i + 0):N_u0 * (i + 1), N_u0 *
                                  (j + 1):N_u0 * (j + 2)]

    elif (N_u1 - j) > (N_u1 - i):
        print('%d, %d: 5' % (i, j))
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] -= cx_arr[N_u0 * (N_u1 - (j + 1)):N_u0 *
                                  (N_u1 - (j + 0)), N_u0 *
                                  (N_u1 - (i + 1)):N_u0 * (N_u1 - (i + 0))]
copy_counter = 0
reuse_counter = 0

for n, (i, j) in enumerate(itertools.product(range(N_u1), repeat=2)):
    if i == 0 and j < 2:
        copy_counter += 1
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] = cx_arr_copy[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
                                      (j + 1)]

    elif i < j and j < (N_u1 - i) and i > 0:
        copy_counter += 1
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 * (j + 1)] = conj(
            cx_arr_copy[N_u0 * j:N_u0 * (j + 1), N_u0 * i:N_u0 * (i + 1)])

    elif i == 0 and j >= 2:
        reuse_counter += 1
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] = cx_arr[N_u0 * 0:N_u0 * (1), N_u0 * (1):N_u0 * (2)]

    elif i == j and i > 0:
        reuse_counter += 1
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 *
               (j + 1)] = cx_arr[N_u0 * 0:N_u0 * (1), N_u0 * (0):N_u0 * (1)]

    elif i > j:
        reuse_counter += 1
        cx_arr[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 * (j + 1)] = conj(
            cx_arr[N_u0 * j:N_u0 * (j + 1), N_u0 * i:N_u0 * (i + 1)])

print(copy_counter)
print(reuse_counter)
print((copy_counter + reuse_counter, N_u1 * N_u1))
"""

#cx_arr_copy -= cx_arr
#cx_arr_copy = abs(cx_arr_copy)
cx_arr = cx_arr[::-1, :]
cx_arr_copy = cx_arr_copy[::-1, :]

u1_arr, u0_arr = meshgrid(u1, u0)

hsv = color_map(cx_arr)
hsv_copy = color_map(cx_arr_copy)

colors_list = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_u1**2)
]

colors = matplotlib.colors.hsv_to_rgb(hsv)
colors_copy = matplotlib.colors.hsv_to_rgb(hsv_copy)

wkwk_arr = array(wkwk_vec).reshape((N_u0 * N_u1, N_u0 * N_u1, 4))
plt.figure(200)
plt.subplot(1, 2, 1)
for n, (i, j) in enumerate(itertools.product(range(N_u1), repeat=2)):
    plt.subplot(N_u1, N_u1, n + 1)

    plt.title('$\omega$: %f' % (wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 * j, 0] -
                                wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 * j, 2]))

    plt.imshow(
        colors[N_u0 * i:N_u0 * (i + 1), N_u0 * j:N_u0 * (j + 1)],
        extent=(0, 1, 0, 1),
    )
"""
plt.imshow(
    colors,
    extent=(0, 1, 0, 1),
)
"""

plt.savefig('plots/potcoef_cx_2d.eps')

plt.figure(300)

plt.imshow(
    colors_copy,
    extent=(0, 1, 0, 1),
)
"""

N_u0_new = 1 << 10
N_u1_new = 1 << 10

u0, du0 = linspace(
    u_max / N_u0_new, 1 - u_max / N_u0_new, N_u0_new, retstep=True)
u1, du1 = linspace(
    -1 + u_max / N_u1_new, 1 - u_max / N_u1_new, N_u1_new, retstep=True)

wk_vec = list(itertools.product(u1 / (1 - u1**2), u0 / (1 - u0)))

t0 = time.time()
green = plasmon_green_v(wk_vec, mu_e, mu_h, sys)

print('[%e], Elapsed: %.2fs' % (mu_e, time.time() - t0))

green_arr = array(green).reshape((N_u1_new, N_u0_new)).T
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

plt.axis([-1, 1, -1, 1])

for n, (i, j) in enumerate(itertools.product(range(N_u1), repeat=2)):
    k_min = abs(wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 * j, 1] -
                wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 * j, 3])

    u0_min = k_min / (1 + k_min)

    k_max = wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 *
                     j, 1] + wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 * j, 3]

    u0_max = k_max / (1 + k_max)

    w0 = wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 *
                  j, 0] - wkwk_arr[N_u0 * (N_u1 - 1 - i), N_u0 * j, 2]

    u1 = (sqrt(1 + 4 * w0**2) - 1) / (2 * w0) if abs(w0) >= 1e-10 else 0

    plt.plot([u0_min, u0_max], [u1, u1],
             '.-',
             color=colors_list[n],
             label='$\omega$: %f' % w0)

plt.legend(loc=0)
"""
plt.show()
