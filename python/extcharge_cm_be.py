from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([3 * 6.8, 5.3]))

n_x, n_y = 4, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]


def load_raw_PL_data(path, eV_max):
    raw = loadtxt(path)
    arg_max = raw[:, 1].argmax()
    xdata_eV = 1240.0 / raw[::-1, 0]
    xdata_eV -= xdata_eV[arg_max]

    xdata_eV_arg = abs(xdata_eV) < eV_max

    return array([
        xdata_eV[xdata_eV_arg],
        raw[xdata_eV_arg, 1] / amax(raw[xdata_eV_arg, 1]),
    ]).T


def adj_r_squared(data, model, n_params=1):
    data_avg = average(data)
    return 1 - sum((model - data)**2) / sum(
        (data - data_avg)**2) * (data.size - n_params - 1) / (data.size - 1)


def aic_criterion(data, model, n_params=1):
    rss = sum((model - data)**2)
    sigma2 = rss / data.size
    return (rss + 2 * n_params * sigma2) / (data.size * sigma2)


def distr_be(E, mu, sys):
    r = (E - mu) * sys.beta
    if abs(r) > 700:
        return 0

    return 1 / (exp(r) - 1)


def distr_mb(E, mu, sys):
    r = -(E - mu) * sys.beta
    if abs(r) > 700:
        return 0

    return exp(r)


def osc_strength(Lx, Ly, nx, ny, sys):
    return Lx * Ly / (nx * ny)**2


def state_energy(Lx, Ly, nx, ny, sys):
    return ((nx / Lx)**2 +
            (ny / Ly)**2) * (pi * sys.c_hbarc)**2 * 0.5 / (sys.m_e + sys.m_h)


def distr_sizes(x, L, hwhm):
    sigma = hwhm / sqrt(2 * log(2))
    return exp(-0.5 * ((x - L) / sigma)**2) / (sigma * sqrt(2 * pi))


def lorentz_distr(E, E0, gamma):
    return 0.5 * gamma / ((E - E0)**2 + 0.25 * gamma**2) / pi


def os_integrand(Lx, Ly, E, gamma, hwhm_x, hwhm_y, nx, ny, sys):
    return osc_strength(Lx, Ly, nx, ny, sys) * distr_sizes(
        Lx, sys.size_Lx, hwhm_x) * distr_sizes(
            Ly, sys.size_Ly, hwhm_y) * lorentz_distr(
                E, state_energy(Lx, Ly, nx, ny, sys), gamma)


def os_integrand_be(Lx, Ly, E, gamma, hwhm_x, hwhm_y, nx, ny, sys):
    E_S = state_energy(Lx, Ly, nx, ny, sys)

    return osc_strength(Lx, Ly, nx, ny, sys) * distr_sizes(
        Lx, sys.size_Lx, hwhm_x) * distr_sizes(
            Ly, sys.size_Ly, hwhm_y) * lorentz_distr(E, E_S, gamma) * distr_mb(
                E_S, 0, sys)


def os_integ(*args):
    return dblquad(
        os_integrand_be,
        0,
        float('inf'),
        0,
        float('inf'),
        args=args,
    )


loaded_data = array([
    load_raw_PL_data('extra/data/extcharge/%s' % f, 0.05)
    for f in os.listdir('extra/data/extcharge')
])
"""
nmax = 9

states_vec = list(
    itertools.product(
        range(1, nmax + 1, 2),
        range(1, nmax + 1, 2),
    ))
"""
states_vec = [
    (1, 1),
    (1, 3),
    (1, 5),
    (3, 1),
    (5, 1),
]

N_E, N_states = 2 << 6, len(states_vec)

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

sys = system_data(m_e, m_h, eps_sol, T, size_d, 1, 1, eps_sol)


def save_PL(ii, sys, pool, size_Lx, size_Ly, hwhm_x, hwhm_y, gamma, shift):
    E_vec = linspace(-0.11, 0.11, N_E)

    sys.size_Lx, sys.size_Ly = size_Lx, size_Ly
    sys.set_hwhm(hwhm_x, hwhm_y)

    starmap_args = [
        (E, gamma, hwhm_x, hwhm_y, nx, ny, sys)
        for (nx, ny), E in itertools.product(states_vec, E_vec - shift)
    ]

    data = array(time_func(
        pool.starmap,
        os_integ,
        starmap_args,
    )).reshape((N_states, N_E, 2))

    starmap_args = [(
        E,
        gamma,
        hwhm_x,
        hwhm_y,
        nx,
        ny,
        sys,
    ) for (nx, ny), E in itertools.product(
        states_vec,
        loaded_data[ii, :, 0] - shift,
    )]

    data_at_fit = array(time_func(
        pool.starmap,
        os_integ,
        starmap_args,
    )).reshape((N_states, loaded_data[ii, :, 0].size, 2))

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/extcharge/cm_be_%s' % file_id,
        [data.flatten()],
        extra_data={
            'size_Lx': size_Lx,
            'size_Ly': size_Ly,
            'hwhm_x': hwhm_x,
            'hwhm_y': hwhm_y,
            'gamma': gamma,
            'shift': shift,
            'states_vec': states_vec,
            'E_vec': E_vec.tolist(),
        },
    )

    save_data(
        'extra/extcharge/cm_be_fit_%s' % file_id,
        [data_at_fit.flatten()],
        extra_data={
            'size_Lx': size_Lx,
            'size_Ly': size_Ly,
            'hwhm_x': hwhm_x,
            'hwhm_y': hwhm_y,
            'gamma': gamma,
            'shift': shift,
            'states_vec': states_vec,
            'E_vec': loaded_data[ii, :, 0].tolist(),
        },
    )

    return file_id


def load_PL(path, file_id):
    extra_dict = {}
    data = load_data(path + ('/cm_be_%s' % file_id), extra_dict)
    data = data.reshape((
        len(extra_dict['states_vec']),
        len(extra_dict['E_vec']),
        2,
    ))

    extra_dict_fit = {}
    data_at_fit = load_data(path + ('/cm_be_fit_%s' % file_id), extra_dict_fit)
    data_at_fit = data_at_fit.reshape((
        len(extra_dict_fit['states_vec']),
        len(extra_dict_fit['E_vec']),
        2,
    ))

    extra_dict['E_vec'] = array(extra_dict['E_vec'])
    extra_dict_fit['E_vec'] = array(extra_dict_fit['E_vec'])

    return (
        extra_dict['E_vec'],
        data,
        extra_dict_fit['E_vec'],
        data_at_fit,
        extra_dict,
        extra_dict_fit,
    )


def plot_PL(ii, sys, pool, data, data_at_fit, extra_dict):
    globals().update(extra_dict)

    sys.size_Lx, sys.size_Ly = size_Lx, size_Ly

    E_avg_vec = E_vec[data[:, :, 0].argmax(axis=1)]

    sum_data = sum(data[:, :, 0], axis=0)
    data[:, :, 0] /= amax(sum_data)
    sum_data /= amax(sum_data)

    sum_data_at_fit = sum(data_at_fit[:, :, 0], axis=0)
    sum_data_at_fit /= amax(sum_data_at_fit)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(states_vec))
    ]

    for data_state, c, E_avg, state in zip(
            data,
            colors,
            E_avg_vec,
            states_vec,
    ):
        ax[ii].plot(
            E_vec,
            data_state[:, 0],
            color=c,
            linewidth=0.7,
            #label=r'$(%d,~%d)$' % tuple(state),
        )

        ax[ii].axvline(
            x=E_avg,
            linestyle='-',
            color=c,
            linewidth=0.5,
        )

    ax[ii].plot(
        E_vec,
        sum_data,
        color='k',
        linewidth=2,
        label='AIC: $%.2f$\nAdj $R^2$: $%.3f$' % (
            aic_criterion(loaded_data[ii, :, 1], sum_data_at_fit, 4),
            adj_r_squared(loaded_data[ii, :, 1], sum_data_at_fit, 4),
        ),
    )

    ax[ii].plot(
        loaded_data[ii, :, 0],
        loaded_data[ii, :, 1],
        marker='o',
        markeredgecolor='m',
        markerfacecolor=(1, 1, 1, 0),
        linestyle='',
    )

    if ii > 0:
        ax[ii].set_yticks([])

    if ii > 0 and ii < 4:
        ax[ii].set_xticklabels(ax[ii].get_xticklabels()[1:-1])

    ax[ii].set_xlim(E_vec[0], E_vec[-1])
    ax[ii].set_ylim(0, None)

    lg = ax[ii].legend(
        loc='upper left',
        title=(r'$%.1f \times %.1f$ nm' % (sys.size_Lx, sys.size_Ly)) + '\n' +
        (r'$\Gamma$: $%.1f$ meV' % (gamma * 1e3)),
        prop={'size': 12},
    )
    lg.get_title().set_fontsize(13)


sizes_vec = [
    (29.32, 5.43),
    (26.11, 6.42),
    (25.40, 8.05),
    (13.74, 13.37),
]
hwhm_vec = [
    (3.3, 0.81),
    (3.34, 1.14),
    (2.9, 0.95),
    (2.17, 1.85),
]
gamma_vec = [
    (0.02275155, -0.01887947),
    (0.02275155, -0.01166034),
    (0.02275155, -0.01008527),
    (0.02275155, -0.00699868),
]

file_id_list = []

pool = multiprocessing.Pool(multiprocessing.cpu_count())

if len(file_id_list) == 0:
    for ii, (
        (Lx, Ly),
        (hwhm_x, hwhm_y),
        (gamma, shift),
    ) in enumerate(zip(sizes_vec, hwhm_vec, gamma_vec)):
        file_id = save_PL(
            ii,
            sys,
            pool,
            Lx,
            Ly,
            hwhm_x,
            hwhm_y,
            gamma,
            shift,
        )
        file_id_list.append(file_id)

print(file_id_list)

for ii, file_id in enumerate(file_id_list):
    _, data, _, data_at_fit, extra_dict, _ = load_PL(
        'extra/extcharge',
        file_id,
    )
    plot_PL(ii, sys, pool, data, data_at_fit, extra_dict)

pool.terminate()

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('cm_be_fit_%d_mb_1g' % len(states_vec)),
    transparent=True,
)

plt.show()
