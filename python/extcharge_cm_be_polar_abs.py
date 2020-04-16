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


def lorentz_cont(energy, gamma_c):
    return 0.5 + arctan(2 * energy / gamma_c) / pi


def load_raw_Abs_data(path, eV_min, eV_max):
    raw = loadtxt(path)
    arg_max = raw[:, 1].argmax()
    xdata_eV = raw[:, 0]
    xdata_eV -= xdata_eV[arg_max]

    xdata_eV_arg = (xdata_eV > -eV_min) * (xdata_eV < eV_max)

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


E_min_data, E_max_data = 0.15, 0.7

labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]

loaded_data = array([
    load_raw_Abs_data(
        'extra/data/extcharge/Abs_%s.txt' % label,
        E_min_data,
        E_max_data,
    ) for label in labels_vec
])

N_E = 1 << 8

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_hh, m_lh, T = 0.27, 0.45, 0.52, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)


def plot_Abs(ii, sys_hh, sys_lh, popt, extra_dict):
    globals().update(extra_dict)

    sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
    sys_hh.set_hwhm(*hwhm_vec[ii])

    sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
    sys_lh.set_hwhm(*hwhm_vec[ii])

    gamma_hh, gamma_lh = popt[:2]
    peak_hh_vec = array(popt[2:6])
    peak_lh_vec = array(popt[6:10])
    gamma_c, energy_c = popt[10:12]
    mag_peak_lh = popt[12]
    mag_cont = popt[13]

    E_vec = linspace(-E_min_data, E_max_data, N_E)
    data_cont = lorentz_cont(E_vec - energy_c, gamma_c) * mag_cont

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(states_vec))
    ]

    data_hh = array([
        exciton_PL_vec(
            E_vec - peak_hh_vec[ii],
            gamma_hh,
            nx,
            ny,
            sys_hh,
        ) for nx, ny in states_vec
    ])

    data_lh = array([
        exciton_PL_vec(
            E_vec - peak_lh_vec[ii],
            gamma_lh,
            nx,
            ny,
            sys_lh,
        ) for nx, ny in states_vec
    ])

    data_hh_sum, data_lh_sum = sum(data_hh, axis=0), sum(data_lh, axis=0)
    data_hh_sum /= amax(data_hh_sum)
    data_lh_sum /= amax(data_lh_sum) / mag_peak_lh

    sum_model = data_hh_sum + data_lh_sum + data_cont
    data_hh_sum /= amax(sum_model)
    data_lh_sum /= amax(sum_model)
    data_cont /= amax(sum_model)
    sum_model /= amax(sum_model)

    ax[ii].plot(
        E_vec,
        data_hh_sum,
        color='r',
        linewidth=0.6,
    )

    ax[ii].plot(
        E_vec,
        data_lh_sum,
        color='b',
        linewidth=0.6,
    )

    ax[ii].plot(
        E_vec,
        data_cont,
        color='m',
        linewidth=0.8,
    )

    ax[ii].plot(
        E_vec,
        sum_model,
        color='k',
        linewidth=2,
    )

    ax[ii].plot(
        loaded_data[ii][:, 0],
        loaded_data[ii][:, 1],
        marker='o',
        markeredgecolor='m',
        markerfacecolor=(1, 1, 1, 0),
        markeredgewidth=1.8,
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
        title=(r'%s: $%.1f \times %.1f$ nm' %
               (labels_vec[ii], sys_hh.size_Lx, sys_lh.size_Ly)),
        prop={'size': 12},
    )
    lg.get_title().set_fontsize(13)


"""
file_id_params = ''

try:
    extra_dict_params = {}
    params_vec = load_data(
        'extra/extcharge/cm_be_polar_fit_params_abs_%s' % file_id_params,
        extra_dict_params,
    )

    gamma_vec = params_vec[:extra_dict_params['n_gamma']]
    shift_vec = params_vec[extra_dict_params['n_gamma']:]
except:
    print('File not found.')
"""

for ii, file_id in enumerate(labels_vec):
    plot_Abs(
        ii,
        sys_hh,
        sys_lh,
        (
            40e-3,
            120e-3,
            -0.03,
            -0.02,
            -0.01,
            0.00,
            0.12,
            0.135,
            0.145,
            0.155,
            80e-3,
            0.43,
            0.75,
            0.7,
        ),
        extra_dict={
            'states_vec':
            list(itertools.product(
                range(1, 5 + 1, 2),
                range(1, 5 + 1, 2),
            )),
            'labels_vec': [
                'BS065',
                'BS006',
                'BS066',
                'BS068',
            ],
            'sizes_vec': [
                (29.32, 5.43),
                (26.11, 6.42),
                (25.4, 8.05),
                (13.74, 13.37),
            ],
            'hwhm_vec': [
                (3.3, 0.81),
                (3.34, 1.14),
                (2.9, 0.95),
                (2.17, 1.85),
            ],
        },
    )

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)
"""
plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('cm_be_polar_%ds_%dg_abs' % (len(states_vec), extra_dict_params['n_gamma'])),
    transparent=True,
)
"""

plt.show()
