from common import *
import pyperclip

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

N_k = 1 << 10

be_exc = -193e-3
be_biexc = -45e-3

surf_area = 326.4  # nm^2
m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 1000  # K
sys = system_data(m_e, m_h, eps_r, T)

mu_e_lim = sys.exc_mu_val(be_exc + 0.5 * be_biexc)
mu_h_lim = sys.get_mu_h(mu_e_lim)
mu_exc_lim = mu_e_lim + mu_h_lim
n_q_inf = sys.density_ideal(mu_e_lim)
n_exc_inf = sys.density_exc(mu_exc_lim, be_exc)

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]


def lambda_th(T, sys):
    return sys.c_hbarc * sqrt(2 * pi / sys.m_p / sys.c_kB / T)


def lambda_th_biexc(T, sys):
    return 2 * sys.c_hbarc * sqrt(pi / (sys.m_e + sys.m_h) / sys.c_kB / T)


def solve_eq_state(n_gamma, sys):
    return result_s(biexciton_eqst_c(
        n_gamma,
        be_exc,
        be_biexc,
        0.0,
        sys,
    )).value


def solve_eqst_data(T_vec, N_vec):
    T_arr, N_arr = meshgrid(T_vec, N_vec)
    N_T_arr = zeros((N_arr.size, 2))
    N_T_arr[:, 0] = N_arr.ravel()
    N_T_arr[:, 1] = T_arr.ravel()

    return array(
        biexciton_eqst_c_vec(
            N_T_arr.ravel(),
            be_exc,
            be_biexc,
            0.0,
            sys,
        )).reshape((N_arr.size, 2))


N_n, N_T = 1 << 9, 1 << 9

n_gamma_vec = logspace(-2.2, 1, N_n) / sys.a0**2
T_vec = logspace(1.8, log10(10e2), N_T)
color_image = zeros((N_T, N_n, 3))

include_free_charges = 1
include_excitons = 1
include_biexcitons = 1

degeneracy = 0

colors = array([
    [0.0, 0.8, 0.8],
    [1 / 3.0, 0.8, 0.8] if not degeneracy else [1 / 3. - 1 / 10.0, 0.8, 0.8],
    [2 / 3.0, 0.8, 0.8] if not degeneracy else [2 / 3. - 1 / 10.0, 0.8, 0.8],
])

colors = array([matplotlib.colors.hsv_to_rgb(c) for c in colors])

mu_e_u_data = solve_eqst_data(T_vec, n_gamma_vec)


def calc_data(enum_n, i, j):
    n_gamma, sys = n_gamma_vec[i], system_data(m_e, m_h, eps_r, T_vec[j])
    mu_e, u = mu_e_u_data[enum_n]

    if isnan(mu_e):
        mu_e = mu_e_lim
        mu_h = mu_h_lim

        n_s_vec = zeros((3, ))
    else:
        mu_h = sys.get_mu_h(mu_e)
        mu_exc = mu_e + mu_h

        n_s_vec = array([
            sys.density_ideal(mu_e),
            sys.density_exc(mu_exc, be_exc),
            sys.density_exc2_u(u),
        ])

    return n_s_vec


def calc_image(
    data,
    i,
    j,
    selection=(0, 1, 2),
    max_data_values=1,
    min_data_values=0,
):
    if degeneracy:
        n_s_vec = (-min_data_values + data[i, j]) / (-min_data_values +
                                                     max_data_values)

    else:
        n_s_vec = data[i, j] / sqrt(sum(data[i, j]**2))

    n_s_vec = array([n_s_vec[i] for i in selection])
    return sum(array([colors[i] for i in selection]).T * n_s_vec, axis=1)


if not include_free_charges and not include_excitons and not include_biexcitons:
    print('Nothing to plot')
    exit()

selector_list = [] + ([0] if include_free_charges else []) + (
    [1] if include_excitons else []) + ([2] if include_biexcitons else [])

data_args = [
    (enum_n, i, j)
    for enum_n, (i, j) in enumerate(itertools.product(range(N_n), range(N_T)))
]

pool = multiprocessing.Pool(multiprocessing.cpu_count())

data_image = array(time_func(
    pool.starmap,
    calc_data,
    data_args,
)).reshape((N_n, N_T, 3))

if degeneracy:
    lambda_th2_arr = repeat(
        lambda_th(T_vec, sys).reshape((1, N_T))**2,
        N_n,
        axis=0,
    ) / 4
    lambda_th_biexc2_arr = repeat(
        lambda_th_biexc(T_vec, sys).reshape((1, N_T))**2,
        N_n,
        axis=0,
    )

    data_image[:, :, 1] *= lambda_th2_arr
    data_image[:, :, 2] *= lambda_th_biexc2_arr
    data_image[:, :, 1] = clip(data_image[:, :, 1], 0, 2)
    data_image[:, :, 2] = clip(data_image[:, :, 2], 0, 2)

    #data_image = log10(data_image)
else:
    data_image *= sys.a0**2

max_data_values = amax(data_image, axis=(0, 1))
min_data_values = amin(data_image, axis=(0, 1))

image_args = [(data_image, i, j, selector_list, max_data_values,
               min_data_values)
              for (i, j) in itertools.product(range(N_n), range(N_T))]

color_image = array(time_func(
    pool.starmap,
    calc_image,
    image_args,
)).reshape((N_n, N_T, 3))

pool.terminate()

n_gamma_vec = n_gamma_vec * sys.a0**2

X, Y = meshgrid(n_gamma_vec, T_vec)

color_tuples = array([
    color_image[:-1, :-1, 0].T.flatten(),
    color_image[:-1, :-1, 1].T.flatten(),
    color_image[:-1, :-1, 2].T.flatten(),
]).transpose()

ax[0].set_xscale('log')
ax[0].set_yscale('log')

im = ax[0].pcolormesh(
    X,
    Y,
    zeros_like(X),
    color=color_tuples,
    snap=True,
    antialiased=True,
    rasterized=True,
)
im.set_array(None)

if len(selector_list) == 1:
    cm = ListedColormap(
        [x * colors[selector_list[0]] for x in linspace(0, 1, 256)])

    min_value, max_value = min_data_values[selector_list[0]], max_data_values[
        selector_list[0]]

    if degeneracy:
        boundaries_list = linspace(0, max_value, 256)
        ticks = floor(linspace(boundaries_list[0], boundaries_list[-1], 6))
        #boundaries_list = linspace(min_value, 1, 256)
        #ticks = linspace(min_value, 0, 5).tolist() + [1]
        format = ['$%.2f$', '$%.0f$', '$%.0f$'][selector_list[0]]
        extend = ['neither', 'max', 'max'][selector_list[0]]
    else:
        boundaries_list = linspace(min_value, max_value, 256)
        ticks = linspace(boundaries_list[0], boundaries_list[-1], 6)
        format = ['$%.2f$', '$%.1f$', '$%.0f$'][selector_list[0]]
        extend = 'neither'

    cb = fig.colorbar(
        ScalarMappable(cmap=cm),
        ax=ax[0],
        boundaries=boundaries_list,
        ticks=ticks,
        format=format,
        fraction=0.05,
        pad=0.01,
        extend=extend,
    )

    if not degeneracy:
        cb.ax.set_ylabel(r'$%s a_0^2$' %
                         ['n_q', 'n_X', 'n_{X_2}'][selector_list[0]])

        cb.ax.yaxis.set_label_coords(1.7, 0.5)

ax[0].plot(
    [6.27 / surf_area * sys.a0**2, 52.85 / surf_area * sys.a0**2],
    [294, 294],
    '-',
    linewidth=0.8,
    color='w',
)

ax[0].plot(
    [6.27 / surf_area * sys.a0**2, 52.85 / surf_area * sys.a0**2],
    [294, 294],
    'o',
    markeredgecolor='w',
    markerfacecolor='#000000',
)
"""
ax[0].axvline(
    x=1,
    color='w',
    linestyle='--',
    dashes=(3., 5.),
    dash_capstyle='round',
    linewidth=0.8,
)
"""

if include_free_charges:
    ax[0].text(
        8e-3,
        9e2,
        r'$q$',
        color='k',
        fontsize=31,
        va='center',
        ha='center',
    )

    ax[0].text(
        8e-3,
        9e2,
        r'$q$',
        color='w',
        fontsize=28,
        va='center',
        ha='center',
    )

if include_excitons:
    ax[0].text(
        4e-2,
        4e2,
        r'$X$',
        color='k',
        fontsize=31,
        va='center',
        ha='center',
    )

    ax[0].text(
        4e-2,
        4e2,
        r'$X$',
        color='w',
        fontsize=28,
        va='center',
        ha='center',
    )

if include_biexcitons:
    ax[0].text(
        3,
        1e2,
        r'$X_2$',
        color='k',
        fontsize=31,
        va='center',
        ha='center',
    )

    ax[0].text(
        3,
        1e2,
        r'$X_2$',
        color='w',
        fontsize=28,
        va='center',
        ha='center',
    )

#ax[0].set_xlabel(r '$\langle N_\gamma \rangle$')
ax[0].set_xlabel(r'$n_\gamma a_0^2$')
ax[0].xaxis.set_label_coords(0.5, -0.05)
ax[0].set_ylabel(r'$T$ (K)')
ax[0].yaxis.set_label_coords(-0.04, 0.5)

print('mu_e_lim: %f, mu_exc_lim: %f' % (mu_e_lim, mu_exc_lim))
print('n_q_inf: %e, n_exc_inf: %e' % (n_q_inf, n_exc_inf))
print('a_0^2: %f nm^2' % sys.a0**2)

plt.tight_layout()
plt.savefig(
    '/storage/Reference/Work/University/PhD/OwnPapers/biexcitons1/figures/%s.pdf'
    % ('biexciton_diagram_%s%s' %
       (''.join(['%s' % i
                 for i in selector_list]), '_deg' if degeneracy else '')),
    transparent=True,
)
plt.show()
