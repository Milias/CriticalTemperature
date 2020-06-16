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


def cou_energy(sys):
    return -sys.c_aEM * sys.c_hbarc / sqrt(sys.size_Lx**2 +
                                           sys.size_Ly**2) / sys.eps_mat


def load_raw_PL_data(path, eV_max):
    raw = loadtxt(path)
    raw[:, 0] = raw[::-1, 0]
    raw[:, 1] = raw[::-1, 1]
    arg_max = raw[:, 1].argmax()
    xdata_eV = 1240.0 / raw[:, 0]
    peak_eV = xdata_eV[arg_max]

    xdata_eV_arg = abs(xdata_eV - peak_eV) < eV_max

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

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]

N_E = 1 << 9

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.52, 0.45, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)


def save_PL(ii, sys, states_vec, size_Lx, size_Ly, hwhm_x, hwhm_y, popt):
    E_vec = linspace(-E_max_data, E_max_data, N_E) + peak_eV_vec[ii]
    gamma_hh = popt[0]
    peak_hh = array(popt[1:5])

    sys.size_Lx, sys.size_Ly = size_Lx, size_Ly
    sys.set_hwhm(hwhm_x, hwhm_y)

    print(time.strftime('%X'))
    print('Γ_hh: %.1f meV' % (gamma_hh * 1e3))
    print('ɛ_hh: %.0f, %.0f, %.0f, %.0f meV' % tuple(peak_hh * 1e3))
    print('\n', flush=True)

    data_hh = array([
        exciton_lorentz_vec(
            E_vec - peak_hh[ii],
            gamma_hh,
            nx,
            ny,
            sys,
        ) for nx, ny in states_vec
    ])

    data_hh_at_fit = array([
        exciton_lorentz_vec(
            loaded_data[ii][:, 0] - peak_hh[ii],
            gamma_hh,
            nx,
            ny,
            sys,
        ) for nx, ny in states_vec
    ])

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/extcharge/cm_be_polar_%s' % file_id,
        [data_hh.flatten()],
        extra_data={
            'size_Lx': size_Lx,
            'size_Ly': size_Ly,
            'hwhm_x': hwhm_x,
            'hwhm_y': hwhm_y,
            'shift': peak_hh.tolist(),
            'gamma': gamma_hh,
            'states_vec': states_vec,
            'E_vec': E_vec.tolist(),
        },
    )

    save_data(
        'extra/extcharge/cm_be_fit_polar_%s' % file_id,
        [data_hh_at_fit.flatten()],
        extra_data={
            'size_Lx': size_Lx,
            'size_Ly': size_Ly,
            'hwhm_x': hwhm_x,
            'hwhm_y': hwhm_y,
            'shift': peak_hh.tolist(),
            'gamma': gamma_hh,
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


def plot_PL(ii, sys, data, data_at_fit, extra_dict, extra_dict_params, popt):
    globals().update(extra_dict)
    gamma_hh = popt[0]
    peak_hh = array(popt[1:5])

    sys.size_Lx, sys.size_Ly = size_Lx, size_Ly

    E_avg_vec = E_vec[data.argmax(axis=1)]

    sum_data = sum(data, axis=0)
    data /= amax(sum_data)
    sum_data /= amax(sum_data)

    sum_data_at_fit = sum(data_at_fit, axis=0)
    sum_data_at_fit /= amax(sum_data_at_fit)

    savetxt(
        'extra/extcharge/export_PL/%s_all.csv' % labels_vec[ii],
        hstack((
            E_vec.T.reshape(-1, 1),
            data.T,
            sum_data.T.reshape(-1, 1),
        )),
        delimiter=',',
        newline='\n',
        header='E_vec (eV), %s, PL (sum)' %
        ', '.join(['%d_%d' % (nx, ny) for nx, ny in states_vec]),
    )

    savetxt(
        'extra/extcharge/export_PL/%s_fit.csv' % labels_vec[ii],
        hstack((
            loaded_data[ii][:, 0].T.reshape(-1, 1),
            data_at_fit.T,
            sum_data_at_fit.T.reshape(-1, 1),
        )),
        delimiter=',',
        newline='\n',
        header='E_vec (eV), %s, PL (sum)' %
        ', '.join(['%d_%d' % (nx, ny) for nx, ny in states_vec]),
    )

    savetxt(
        'extra/extcharge/export_PL/%s_exp.csv' % labels_vec[ii],
        hstack((loaded_data[ii][:, 0].T.reshape(-1, 1),
                loaded_data[ii][:, 1].T.reshape(-1, 1))),
        delimiter=',',
        newline='\n',
        header='E_vec (eV), PL (sum)',
    )

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(states_vec))
    ]

    for jj, (
            data_state,
            E_avg,
            state,
    ) in enumerate(zip(
            data,
            E_avg_vec,
            states_vec,
    )):
        ax[ii].plot(
            E_vec,
            data_state,
            color=colors[jj],
            linewidth=0.9,
            linestyle='--',
        )

    ax[ii].plot(
        E_vec,
        sum_data,
        color='k',
        linewidth=1.8,
        label='AIC: $%.3f$\nAdj $R^2$: $%.3f$' % (
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
        ax[ii].set_xticklabels([])

    ax[ii].set_xlim(E_vec[0], E_vec[-1])
    ax[ii].set_ylim(0, 1.2)

    lg = ax[ii].legend(
        loc='upper left',
        title=(r'%s: $%.1f \times %.1f$ nm' %
               (labels_vec[ii], sys.size_Lx, sys.size_Ly)) +
        ('\n' +
         (r'$\epsilon_{hh}$: $%.0f\pm%.1f$ meV' %
          (peak_hh[ii] * 1e3,
           sqrt(extra_dict_params['pcov'][ii + 1][ii + 1]) * 1e3)) + '\n' +
         (r'$\Gamma$: $%.1f\pm%.1f$ meV' %
          (gamma * 1e3, sqrt(extra_dict_params['pcov'][0][0]) * 1e3)))
        if gamma else '',
        prop={'size': 12},
    )
    lg.get_title().set_fontsize(14)


#file_id_params = 'MxJ9JSiIQTGfQirOhPOEcg'
#file_id_params = 'nWdmSwy9QnCtQvySivCvUw'
#file_id_params = 'B9lXdF7lRv-t6I4J-oJffQ'
file_id_params = 'd-fccRMqSQKcwq3ju3MHtw'
#file_id_params = '7Vb0FoIESG-25gl1DKjWBw'
#file_id_params = '8BmZaec0QuysuEPFplSkXA'

extra_dict_params = {}
popt = load_data(
    'extra/extcharge/cm_be_polar_fit_params_%s' % file_id_params,
    extra_dict_params,
)

savetxt(
    'extra/extcharge/export_PL/popt.csv',
    array(popt).reshape((1, -1)),
    delimiter=',',
    newline='\n',
    header='gamma_hh (eV), %s' %
    ', '.join(['peak_hh (%s) (eV)' % l for l in labels_vec]),
)

savetxt(
    'extra/extcharge/export_PL/pcov.csv',
    array(extra_dict_params['pcov']),
    delimiter=',',
    newline='\n',
    header='gamma_hh (eV), %s' %
    ', '.join(['peak_hh (%s) (eV)' % l for l in labels_vec]),
)

print(popt)
"""
file_id_list = [
    'KzrrIRvPRr-hk1Xl6-y37A',
    't8SPJrOGSFaU8SAufp9AoQ',
    'GoMgpNIfSxGuCrcrSbDOKw',
    'pA7ZCbM4TMe4RXsfFPdfMA',
]

file_id_list = [
    'XqSCXRjKTH6HFASVYV9AaA',
    '0ouYQitlRlacqCPXaCwPCw',
    'iEVCF_CNSn2EV7-ELQNkOQ',
    'c25wm-qVR7-hk6CUTHLWTw',
]
file_id_list = ['bo3FNjx-SgObbIbgDx3jYQ', 'CKTZcRisTq-fcs_UDjmrIw', 'ETVBWXS6RF2ajeT5k1TcKQ', 'xpat21uVQIi0BUBK5GmA6g']
"""
file_id_list = []

if len(file_id_list) == 0:
    for ii, ((Lx, Ly), (hwhm_x, hwhm_y)) in enumerate(
            itertools.zip_longest(
                extra_dict_params['sizes_vec'],
                extra_dict_params['hwhm_vec'],
            )):

        file_id = save_PL(
            ii,
            sys_hh,
            extra_dict_params['states_vec'],
            Lx,
            Ly,
            hwhm_x,
            hwhm_y,
            popt,
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
        sys_hh,
        data,
        data_at_fit,
        extra_dict,
        extra_dict_params,
        popt,
    )

ax[0].set_xlabel(r'$\epsilon$ (eV)')

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('cm_be_polar_%ds_%dp' %
     (len(states_vec), len(diag(extra_dict_params['pcov'])))),
    transparent=True,
)

plt.show()
