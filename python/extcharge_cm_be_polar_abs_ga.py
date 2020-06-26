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


def cont_func(energy, gamma_c, sys):
    return array(exciton_cont_ga_vec(energy, gamma_c, sys))


def load_raw_Abs_data(path, eV_min, eV_max):
    raw = loadtxt(path)
    arg_max = raw[:, 1].argmax()
    xdata_eV = raw[:, 0]
    peak_eV = xdata_eV[arg_max]

    xdata_eV_arg = ((xdata_eV - peak_eV) > -eV_min) * (
        (xdata_eV - peak_eV) < eV_max)

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


file_id_PL = 'M_NdmTxjQSezNji-P10AHw'
params_PL_dict = {}
popt_PL = load_data(
    'extra/extcharge/cm_be_polar_fit_params_ga_%s' % file_id_PL,
    params_PL_dict,
)

print(popt_PL.tolist(), flush=True)

globals().update(params_PL_dict)

nmax = 5

states_vec = list(
    itertools.product(
        range(1, nmax + 1, 2),
        range(1, nmax + 1, 2),
    ))

N_E = 1 << 8
N_states = len(states_vec)
N_samples = len(sizes_vec)

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.52, 0.45, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)

E_min_data, E_max_data = 0.1, 0.65

loaded_data = array([
    load_raw_Abs_data(
        'extra/data/extcharge/Abs_%s.txt' % label,
        E_min_data,
        E_max_data,
    ) for label in labels_vec
])

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]


def plot_Abs(ii, sys_hh, sys_lh, popt, extra_dict):
    globals().update(extra_dict)
    E_vec = linspace(-E_min_data, E_max_data, N_E) + peak_eV_vec[ii]

    sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
    sys_hh.set_hwhm(*hwhm_vec[ii])

    sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
    sys_lh.set_hwhm(*hwhm_vec[ii])

    pcov = array(extra_dict.get('pcov', zeros((popt.size, popt.size))))
    perr = sqrt(diag(pcov))

    #args : (20)
    #mag_peak_lh(4), mag_cont_lh(4), mag_cont_hh(4)
    #E_hh[ <0], E_lh[ <0], energy_c_hh[> 0], energy_c_lh[> 0]
    #gamma_hh, gamma_lh, gamma_c_hh, gamma_c_lh
    mag_peak_lh_vec = popt[:4]
    mag_cont_lh_vec = popt[4:8]
    mag_cont_hh_vec = popt[8:12]
    E_hh, E_lh = array(popt[12:16]), array(popt[16:20])
    energy_c_hh, energy_c_lh = array(popt[20:24]), array(popt[24:28])
    gamma_hh, gamma_lh, gamma_c_hh, gamma_c_lh = popt_PL[0], *popt[28:31]

    E_hh_err, E_lh_err = perr[12:16], perr[16:20]

    print(time.strftime('%X'))

    print('Γ_hh: %.1f meV, Γ_lh: %.1f meV' % (
        gamma_hh * 1e3,
        gamma_lh * 1e3,
    ))
    print('E_hh: %.1f, %.1f, %.1f, %.1f meV' % tuple(E_hh * 1e3))
    print('E_lh: %.1f, %.1f, %.1f, %.1f meV' % tuple(E_lh * 1e3))
    print('Γ_c_hh: %.0f meV, ɛ_c_hh: %.0f, %.0f, %.0f, %.0f meV' % (
        gamma_c_hh * 1e3,
        *tuple(energy_c_hh * 1e3),
    ))
    print('Γ_c_lh: %.0f meV, ɛ_c_lh: %.0f, %.0f, %.0f, %.0f meV' % (
        gamma_c_lh * 1e3,
        *tuple(energy_c_lh * 1e3),
    ))
    print('mag_peak_lh: %.2f, %.2f, %.2f, %.2f' % tuple(mag_peak_lh_vec))
    print('mag_cont_lh: %.2f, %.2f, %.2f, %.2f' % tuple(mag_cont_lh_vec))
    print('mag_cont_hh: %.2f, %.2f, %.2f, %.2f' % tuple(mag_cont_hh_vec))
    print('\n', flush=True)

    sum_model_all = []

    data_cont_hh = cont_func(
        E_vec - energy_c_hh[ii],
        gamma_c_hh,
        sys_hh,
    ) * mag_cont_hh_vec[ii]

    data_cont_lh = cont_func(
        E_vec - energy_c_lh[ii],
        gamma_c_lh,
        sys_lh,
    ) * mag_cont_lh_vec[ii]

    data_hh = array([
        exciton_ga_nomb_vec(
            E_vec - (energy_c_hh[ii] + E_hh[ii]),
            gamma_hh,
            nx,
            ny,
            sys_hh,
        ) for nx, ny in states_vec
    ])

    data_lh = array([
        exciton_ga_nomb_vec(
            E_vec - (energy_c_lh[ii] + E_lh[ii]),
            gamma_lh,
            nx,
            ny,
            sys_lh,
        ) for nx, ny in states_vec
    ])

    data_hh_sum, data_lh_sum = sum(data_hh, axis=0), sum(data_lh, axis=0)

    data_hh /= amax(data_hh_sum)
    data_lh /= amax(data_lh_sum) / mag_peak_lh_vec[ii]
    data_hh_sum /= amax(data_hh_sum)
    data_lh_sum /= amax(data_lh_sum) / mag_peak_lh_vec[ii]

    sum_model = data_hh_sum + data_lh_sum + data_cont_hh + data_cont_lh

    data_hh /= amax(sum_model)
    data_lh /= amax(sum_model)
    data_hh_sum /= amax(sum_model)
    data_lh_sum /= amax(sum_model)
    data_cont_hh /= amax(sum_model)
    data_cont_lh /= amax(sum_model)
    sum_model /= amax(sum_model)

    sum_model_interp = interp1d(E_vec, sum_model)
    sum_model_at_fit = sum_model_interp(loaded_data[ii][:, 0])

    savetxt(
        'extra/extcharge/export_abs/%s_ga_states.csv' % labels_vec[ii],
        hstack((
            E_vec.T.reshape(-1, 1),
            data_hh.T,
            data_lh.T,
        )),
        delimiter=',',
        newline='\n',
        header='E_vec (eV), states_hh %s, states_lh %s' %
        tuple(2 * [','.join(['%d_%d' % (nx, ny) for nx, ny in states_vec])]),
    )

    savetxt(
        'extra/extcharge/export_abs/%s_ga_sum.csv' % labels_vec[ii],
        hstack((
            E_vec.T.reshape(-1, 1),
            data_hh_sum.T.reshape(-1, 1),
            data_lh_sum.T.reshape(-1, 1),
            data_cont_hh.T.reshape(-1, 1),
            data_cont_lh.T.reshape(-1, 1),
            sum_model.T.reshape(-1, 1),
        )),
        delimiter=',',
        newline='\n',
        header=
        'E_vec (eV),data_hh (sum),data_lh (sum),cont_hh,cont_lh,abs (sum)',
    )

    savetxt(
        'extra/extcharge/export_abs/%s_exp.csv' % labels_vec[ii],
        hstack((loaded_data[ii][:, 0].T.reshape(-1, 1),
                loaded_data[ii][:, 1].T.reshape(-1, 1))),
        delimiter=',',
        newline='\n',
        header='E_vec (eV), abs (sum)',
    )

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, N_states * 2 + 2)
    ]

    for jj, (d_hh, d_lh) in enumerate(zip(data_hh, data_lh)):
        ax[ii].plot(
            E_vec,
            d_hh,
            color=colors[jj + 1],
            linestyle='--',
            linewidth=0.5,
        )
        ax[ii].plot(
            E_vec,
            d_lh,
            color=colors[-jj - 2],
            linestyle='--',
            linewidth=0.5,
        )

    ax[ii].plot(
        E_vec,
        data_hh_sum,
        color=colors[0],
        linewidth=0.9,
    )

    ax[ii].plot(
        E_vec,
        data_lh_sum,
        color=colors[-1],
        linewidth=0.9,
    )

    ax[ii].plot(
        E_vec,
        data_cont_hh,
        color='m',
        linewidth=0.9,
    )

    ax[ii].plot(
        E_vec,
        data_cont_lh,
        color='g',
        linewidth=0.9,
    )
    """
    label=(r'$\epsilon_{lh}$: $%.0f\pm%.0f$ meV' % (
        peak_lh_vec[ii] * 1e3,
        perr[6 + ii] * 1e3,
    )) + '\n' + (r'$\Gamma_{lh}$: $%.0f\pm%.0f$ meV' % (
        gamma_lh * 1e3,
        perr[1] * 1e3,
    )),
    """
    """
    label=(r'$\epsilon_{c}$: $%.0f\pm%.0f$ meV' % (
        energy_c[ii] * 1e3,
        perr[11 + ii] * 1e3,
    )) + '\n' + (r'$\Gamma_{c}$: $%.0f\pm%.0f$ meV' % (
        gamma_c * 1e3,
        perr[10] * 1e3,
    )),
    """

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

    ax[ii].set_xlim(E_vec[0], E_vec[-1])
    ax[ii].set_ylim(0, None)

    if ii > 0:
        ax[ii].set_yticks([])
        ax[ii].set_xticklabels([])

    lg = ax[ii].legend(
        loc='upper right',
        title=(r'%s: $%.1f \times %.1f$ nm' %
               (labels_vec[ii], sys_hh.size_Lx, sys_lh.size_Ly)) + '\n' +
        ('AIC: $%.3f$, Adj $R^2$: $%.3f$\n' % (
            aic_criterion(loaded_data[ii][:, 1], sum_model_at_fit, len(popt)),
            adj_r_squared(loaded_data[ii][:, 1], sum_model_at_fit, len(popt)),
        )) + ('$E_{hh}$: $%.0f\pm%.0f$ meV\n$E_{lh}$: $%.0f\pm%.0f$ meV' %
              (E_hh[ii] * 1e3, E_hh_err[ii] * 1e3, E_lh[ii] * 1e3,
               E_lh_err[ii] * 1e3)),
        prop={'size': 11},
    )
    lg.get_title().set_fontsize(11)


extra_dict_params = params_PL_dict
del extra_dict_params['pcov']
file_id_params = 'evPVjZVzR8iqE7cam0GKkg'

popt = load_data(
    'extra/extcharge/cm_be_polar_fit_params_abs_ga_%s' % file_id_params,
    extra_dict_params,
)
"""
popt = array([
    #
    0.79,
    0.72,
    0.71,
    0.81,
    #
    0.35,
    0.35,
    0.35,
    0.39,
    #
    0.32,
    0.27,
    0.26,
    0.30,
    #
    -266.5e-3,
    -268.5e-3,
    -276.8e-3,
    -283.1e-3,
    #
    -258.4e-3,
    -254.6e-3,
    -245.0e-3,
    -243.4e-3,
    #
    2679e-3,
    2677e-3,
    2687e-3,
    2694e-3,
    #
    2819e-3,
    2813e-3,
    2810e-3,
    2812e-3,
    #
    79.5e-3,
    57e-3,
    119e-3,
])

extra_dict_params = {'pcov': zeros((popt.size, popt.size))}
"""

print(popt.tolist(), flush=True)

savetxt(
    'extra/extcharge/export_abs/ga_popt.csv',
    array(popt).reshape((1, -1)),
    delimiter=',',
    newline='\n',
    header=(','.join(['mag_peak_lh (%s)' % l for l in labels_vec])) + ',' +
    (','.join(['mag_cont_lh (%s)' % l for l in labels_vec])) + ',' +
    (','.join(['mag_cont_hh (%s)' % l for l in labels_vec])) + ',' +
    (','.join(['E_hh (%s) (eV)' % l for l in labels_vec])) + ',' +
    (','.join(['E_lh (%s) (eV)' % l for l in labels_vec])) + ',' +
    (','.join(['E_c_hh (%s) (eV)' % l for l in labels_vec])) + ',' +
    (','.join(['E_c_lh (%s) (eV)' % l for l in labels_vec])) + ',' +
    'gamma_lh (eV),gamma_c_hh (eV),gamma_c_lh (eV)',
)

savetxt(
    'extra/extcharge/export_abs/ga_pcov.csv',
    array(extra_dict_params['pcov']),
    delimiter=',',
    newline='\n',
    header=(','.join(['mag_peak_lh (%s)' % l for l in labels_vec])) + ',' +
    (','.join(['mag_cont_lh (%s)' % l for l in labels_vec])) + ',' +
    (','.join(['mag_cont_hh (%s)' % l for l in labels_vec])) + ',' +
    (','.join(['E_hh (%s) (eV)' % l for l in labels_vec])) + ',' +
    (','.join(['E_lh (%s) (eV)' % l for l in labels_vec])) + ',' +
    (','.join(['E_c_hh (%s) (eV)' % l for l in labels_vec])) + ',' +
    (','.join(['E_c_lh (%s) (eV)' % l for l in labels_vec])) + ',' +
    'gamma_lh (eV),gamma_c_hh (eV),gamma_c_lh (eV)',
)

ax[0].set_xlabel(r'$\epsilon$ (eV)')

for ii, file_id in enumerate(labels_vec):
    plot_Abs(ii, sys_hh, sys_lh, popt, extra_dict_params)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('cm_be_polar_%ds_abs_%dp_ga' % (len(states_vec), popt.size)),
    transparent=True,
)

plt.show()
