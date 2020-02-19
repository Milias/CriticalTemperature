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
m_e, m_h, eps_r, T = 0.27, 0.45, 6.369171898453055, 1000  # K
sys = system_data(m_e, m_h, eps_r, T)

mu_e_lim = sys.exc_mu_val(be_exc + 0.5 * be_biexc)
mu_h_lim = sys.get_mu_h(mu_e_lim)
mu_exc_lim = mu_e_lim + mu_h_lim
n_q_inf = sys.density_ideal(mu_e_lim)
n_exc_inf = sys.density_exc(mu_exc_lim, be_exc)

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 2
fig, ax = plt.subplots(n_x, n_y, sharey=True, figsize=fig_size)


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
#N_n, N_T = 1 << 4, 1 << 4

n_gamma_vec = logspace(-2.2, 1, N_n) / sys.a0**2
T_vec = logspace(1.8, log10(10e2), N_T)
color_image = zeros((N_T, N_n, 3))

try:
    argv_data = json.loads(pysys.argv[-1])
except:
    argv_data = {}

include_free_charges = argv_data.get('include_free_charges', 1)
include_excitons = argv_data.get('include_excitons', 1)
include_biexcitons = argv_data.get('include_biexcitons', 1)

show_fig = argv_data.get('show_fig', 1)
file_id = argv_data.get('file_id', '')

savefig_folder = 'plots/papers/biexciton1'

hsv_colors = array([
    [0.0, 0.8, 0.8],
    [1 / 3.0, 0.8, 0.8],
    [2 / 3.0, 0.8, 0.8],
])

hsv_colors_total = array([
    [0.05, 0.8, 0.8],
    [1 / 3.0 + 0.05, 0.8, 0.8],
    [2 / 3.0 + 0.05, 0.8, 0.8],
])

colors = array([matplotlib.colors.hsv_to_rgb(c) for c in hsv_colors])
colors_total = array(
    [matplotlib.colors.hsv_to_rgb(c) for c in hsv_colors_total])

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
    norm=0,
    selection=(0, 1, 2),
    max_data_values=1,
    min_data_values=0,
):
    if norm == 0:
        result = data[i, j] / sqrt(sum(data[i, j]**2))
    elif norm == 1:
        result = data[i, j] / max_data_values

    result = clip(
        sum(array([result[i] * colors[i] for i in selection]), axis=0),
        0,
        1,
    )

    return result


if not include_free_charges and not include_excitons and not include_biexcitons:
    print('Nothing to plot')
    exit()

selector_list = [] + ([0] if include_free_charges else []) + (
    [1] if include_excitons else []) + ([2] if include_biexcitons else [])

if len(selector_list) > 1:
    print('Too many plots')
    exit()

data_args = [
    (enum_n, i, j)
    for enum_n, (i, j) in enumerate(itertools.product(range(N_n), range(N_T)))
]

n_gamma_vec *= sys.a0**2
X, Y = meshgrid(n_gamma_vec, T_vec)

pool = multiprocessing.Pool(multiprocessing.cpu_count())

if file_id == '':
    data_image = array(time_func(
        pool.starmap,
        calc_data,
        data_args,
    )).reshape((N_n, N_T, 3))

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/biexcitons/density_2d_%s' % file_id,
        [
            data_image.flatten(),
        ],
    )
else:
    data_image = load_data(
        'extra/biexcitons/density_2d_%s' % file_id,
        globals(),
    ).reshape((N_n, N_T, 3))

data_image *= sys.a0**2

max_data_values = amax(data_image, axis=(0, 1))
min_data_values = amin(data_image, axis=(0, 1))

image_args = [(
    data_image,
    i,
    j,
    0,
    selector_list,
    max_data_values,
    min_data_values,
) for (i, j) in itertools.product(range(N_n), range(N_T))]

image_args_total = [(
    data_image,
    i,
    j,
    1,
    selector_list,
    max_data_values,
    min_data_values,
) for (i, j) in itertools.product(range(N_n), range(N_T))]

color_image = array(time_func(
    pool.starmap,
    calc_image,
    image_args,
)).reshape((N_n, N_T, 3))

color_image_total = array(
    time_func(
        pool.starmap,
        calc_image,
        image_args_total,
    )).reshape((N_n, N_T, 3))

color_tuples = array([
    color_image[:-1, :-1, 0].T.flatten(),
    color_image[:-1, :-1, 1].T.flatten(),
    color_image[:-1, :-1, 2].T.flatten(),
]).transpose()

color_tuples_total = array([
    color_image_total[:-1, :-1, 0].T.flatten(),
    color_image_total[:-1, :-1, 1].T.flatten(),
    color_image_total[:-1, :-1, 2].T.flatten(),
]).transpose()

pool.terminate()

for axis in ax:
    axis.set_xscale('log')
    axis.set_yscale('log')

n_boundaries = 11
n_ticks = 4

cm = ListedColormap(
    [x * colors[selector_list[0]] for x in linspace(0, 1, 256)])

max_value_total = amax(data_image[:, :, selector_list[0]])
min_value_total = amin(data_image[:, :, selector_list[0]])

max_value = amax(data_image[:, :, selector_list[0]] /
                 sqrt(sum(data_image**2, axis=2)))
min_value = amin(data_image[:, :, selector_list[0]] /
                 sqrt(sum(data_image**2, axis=2)))

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

im_total = ax[0].pcolormesh(
    X,
    Y,
    data_image[:, :, selector_list[0]].T,
    cmap=cm,
    norm=BoundaryNorm(
        linspace(min_value, max_value, n_boundaries),
        ncolors=cm.N,
        clip=True,
    ),
    antialiased=True,
    rasterized=True,
)

boundaries_list = linspace(0, max_value, 256)
ticks = linspace(0, max_value, n_ticks)
format = ['$%.2f$', '$%.1f$', '$%.1f$'][selector_list[0]]

boundaries_list_total = linspace(0, max_value, n_boundaries)
format_total = ['$%.2f$', '$%.1f$', '$%.0f$'][selector_list[0]]

cb = fig.colorbar(
    ScalarMappable(cmap=cm),
    ax=ax[0],
    boundaries=boundaries_list,
    ticks=ticks,
    format=format,
    fraction=0.05,
    pad=0.01,
    drawedges=False,
    orientation='horizontal',
)

cb_total = fig.colorbar(
    ScalarMappable(cmap=cm),
    ax=ax[1],
    boundaries=boundaries_list_total,
    ticks=ticks,
    format=format_total,
    fraction=0.05,
    pad=0.01,
    drawedges=False,
    orientation='horizontal',
)

species_label = ['q', 'X', '{X_2}'][selector_list[0]]

cb.ax.set_xlabel(r'$n_%s a_0^2$' % species_label)
cb.ax.xaxis.set_label_coords(0.5, 0.5)
cb_total.ax.set_xlabel(r'$\mathcal{X}_%s$' % species_label)
cb_total.ax.xaxis.set_label_coords(0.5, -0.5)

for axis in ax:
    axis.plot(
        [6.27 / surf_area * sys.a0**2, 52.85 / surf_area * sys.a0**2],
        [294, 294],
        '-',
        linewidth=0.8,
        color='w',
    )

    axis.plot(
        [6.27 / surf_area * sys.a0**2, 52.85 / surf_area * sys.a0**2],
        [294, 294],
        'o',
        markeredgecolor='w',
        markerfacecolor='#000000',
    )

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

for axis in ax:
    axis.xaxis.tick_top()
    axis.set_xlabel(r'$n_\gamma a_0^2$')
    axis.xaxis.set_label_coords(0.5, 1.07)

ax[0].set_ylabel(r'$T$ (K)')
ax[0].yaxis.set_label_coords(-0.04, 0.5)

plt.tight_layout()
fig.subplots_adjust(wspace=0)

plt.savefig(
    '%s/%s.pdf' % (savefig_folder, 'biexciton_diagram_%s_dual_C1' % (
        ''.join(['%s' % i for i in selector_list]),
    )),
    transparent=True,
)

if show_fig:
    plt.show()
