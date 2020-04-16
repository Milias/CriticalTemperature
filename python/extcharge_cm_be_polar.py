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


E_max_data = 0.15

labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]

loaded_data = array([
    load_raw_PL_data('extra/data/extcharge/PL-%s.txt' % label, E_max_data)
    for label in labels_vec
])

N_E = 1 << 9

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

sys = system_data(m_e, m_h, eps_sol, T, size_d, 1, 1, eps_sol)


def save_PL(ii,
            sys,
            states_vec,
            size_Lx,
            size_Ly,
            hwhm_x,
            hwhm_y,
            shift,
            gamma=None):
    E_vec = linspace(-E_max_data, E_max_data, N_E)

    sys.size_Lx, sys.size_Ly = size_Lx, size_Ly
    sys.set_hwhm(hwhm_x, hwhm_y)

    if gamma is None:
        data = array([
            exciton_PL_d_vec(E_vec - shift, nx, ny, sys)
            for nx, ny in states_vec
        ])

        data_at_fit = array([
            exciton_PL_d_vec(loaded_data[ii][:, 0] - shift, nx, ny, sys)
            for nx, ny in states_vec
        ])
    else:
        data = array([
            exciton_PL_vec(E_vec - shift, gamma, nx, ny, sys)
            for nx, ny in states_vec
        ])

        data_at_fit = array([
            exciton_PL_vec(loaded_data[ii][:, 0] - shift, gamma, nx, ny, sys)
            for nx, ny in states_vec
        ])

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/extcharge/cm_be_polar_%s' % file_id,
        [data.flatten()],
        extra_data={
            'size_Lx': size_Lx,
            'size_Ly': size_Ly,
            'hwhm_x': hwhm_x,
            'hwhm_y': hwhm_y,
            'shift': shift,
            'gamma': gamma,
            'states_vec': states_vec,
            'E_vec': E_vec.tolist(),
        },
    )

    save_data(
        'extra/extcharge/cm_be_fit_polar_%s' % file_id,
        [data_at_fit.flatten()],
        extra_data={
            'size_Lx': size_Lx,
            'size_Ly': size_Ly,
            'hwhm_x': hwhm_x,
            'hwhm_y': hwhm_y,
            'shift': shift,
            'gamma': gamma,
            'states_vec': states_vec,
            'E_vec': loaded_data[ii][:, 0].tolist(),
        },
    )

    return file_id


def load_PL(path, file_id):
    extra_dict = {}
    data = load_data(path + ('/cm_be_polar_%s' % file_id), extra_dict)
    data = data.reshape((
        len(extra_dict['states_vec']),
        len(extra_dict['E_vec']),
    ))

    extra_dict_fit = {}
    data_at_fit = load_data(path + ('/cm_be_fit_polar_%s' % file_id),
                            extra_dict_fit)
    data_at_fit = data_at_fit.reshape((
        len(extra_dict_fit['states_vec']),
        len(extra_dict_fit['E_vec']),
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


def plot_PL(ii, sys, data, data_at_fit, extra_dict, extra_dict_params):
    globals().update(extra_dict)

    sys.size_Lx, sys.size_Ly = size_Lx, size_Ly

    E_avg_vec = E_vec[data.argmax(axis=1)]

    sum_data = sum(data, axis=0)
    data /= amax(sum_data)
    sum_data /= amax(sum_data)

    sum_data_at_fit = sum(data_at_fit, axis=0)
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
            data_state,
            color=c,
            linewidth=0.7,
            #label = r '$(%d,~%d)$' % tuple(state),
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
            aic_criterion(loaded_data[ii][:, 1], sum_data_at_fit, 4),
            adj_r_squared(loaded_data[ii][:, 1], sum_data_at_fit, 4),
        ),
    )

    ax[ii].plot(
        loaded_data[ii][:, 0],
        loaded_data[ii][:, 1],
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
        title=(r'%s: $%.1f \times %.1f$ nm' %
               (labels_vec[ii], sys.size_Lx, sys.size_Ly)) +
        ('\n' + (r'$\Gamma$: $%.1f\pm%.1f$ meV' %
                 (gamma * 1e3, sqrt(extra_dict_params['pcov'][0][0]) * 1e3)))
        if gamma else '',
        prop={'size': 12},
    )
    lg.get_title().set_fontsize(13)


#file_id_params = 'G5Hij-ziSzyL5Ag_qBoijA'
file_id_params = 'xOuG8mm8To20oIYcnfQf2A'
#file_id_params = '1WU-Ie53SPOvsRVDFW5brw'

file_id_params = 'zGccCH6AQri1JT7_w-o5Zg'

extra_dict_params = {}
params_vec = load_data(
    'extra/extcharge/cm_be_polar_fit_params_%s' % file_id_params,
    extra_dict_params,
)

gamma_vec = params_vec[:extra_dict_params['n_gamma']]
shift_vec = params_vec[extra_dict_params['n_gamma']:]

file_id_list = [
    'J2anlScRTSOAGgfGS2xNqw',
    '3PABTvKlS8y5EMe8fY98hw',
    'ks8Fj3jpQY-sztLBua40bg',
    'pyzZTCqyQwm58YmNo0gB1Q',
]
"""
file_id_list = [
    'Y7vNA7D_THmtuFZTURvopw',
    'LxKeZw0XSwav9zQ2w9NiEA',
    'CpTfNtItTnmEno70rji7Eg',
    'vYbyJMkYTfimniHO7gRMRw',
]
file_id_list = [
    '-GtwTCTISm-qJ6-wHOEoFw',
    'soZ_ZT3cRCSH2c3n4J4JHw',
    'xYAw4XglQmSJYDHKBIg5uw',
    'cG9gcScFS8GZ4QJ9GjCn0g',
]
file_id_list = [
    'DIXL_2vnStS8NVa8bvyMiw',
    '_jEmNAn_TGaeGPwaWsIC5A',
    'XQ5jAKDoT4meealwuI9ZCg',
    '8aFXef3vSSawEVC7EGB6VA',
]
"""

if len(file_id_list) == 0:
    for ii, ((Lx, Ly), (hwhm_x, hwhm_y), gamma, shift) in enumerate(
            itertools.zip_longest(
                extra_dict_params['sizes_vec'],
                extra_dict_params['hwhm_vec'],
                gamma_vec,
                shift_vec,
                fillvalue=gamma_vec[-1] if len(gamma_vec) > 0 else None,
            )):

        file_id = save_PL(
            ii,
            sys,
            extra_dict_params['states_vec'],
            Lx,
            Ly,
            hwhm_x,
            hwhm_y,
            shift,
            gamma,
        )
        file_id_list.append(file_id)

print(file_id_list)

for ii, file_id in enumerate(file_id_list):
    _, data, _, data_at_fit, extra_dict, _ = load_PL(
        'extra/extcharge',
        file_id,
    )
    plot_PL(
        ii,
        sys,
        data,
        data_at_fit,
        extra_dict,
        extra_dict_params,
    )

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('cm_be_polar_%ds_%dg' % (len(states_vec), extra_dict_params['n_gamma'])),
    transparent=True,
)

plt.show()
