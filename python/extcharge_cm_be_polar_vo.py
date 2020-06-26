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


labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]

N_samples = len(labels_vec)
E_max_data = 0.15

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


file_id_params = 'rBYyUhVlQ327WuiwjHkVhQ'

extra_dict_params = {}
popt = load_data(
    'extra/extcharge/cm_be_polar_fit_params_vo_%s' % file_id_params,
    extra_dict_params,
)

file_id_params_abs = '0NCcd7uWQOqeUUptm6rk9A'

extra_dict_params_abs = {}
popt_abs = load_data(
    'extra/extcharge/cm_be_polar_fit_params_abs_vo_%s' % file_id_params_abs,
    extra_dict_params_abs,
)

print(popt.tolist(), flush=True)
print(popt_abs.tolist(), flush=True)

max_state = 8

def plot_PL(ii, sys_hh, sys_lh, popt, extra_dict):
    globals().update(extra_dict)

    states_hh_vec = [states_sorted_os(max_state, Lx, Ly) for Lx, Ly in sizes_vec]
    states_lh_vec = [states_sorted_os(max_state, Lx, Ly) for Lx, Ly in sizes_vec]

    gamma, sigma = popt[0], popt[1]
    peak_hh = array(popt[2:6])

    E_vec = linspace(-E_max_data, E_max_data, N_E) + peak_eV_vec[ii]
    gamma_hh, sigma_hh = popt[0], popt[1]
    peak_hh = array(popt[2:6])

    sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
    sys_hh.set_hwhm(*hwhm_vec[ii])

    sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
    sys_lh.set_hwhm(*hwhm_vec[ii])

    print(time.strftime('%X'))
    print('Γ_hh: %.1f meV' % (gamma_hh * 1e3))
    print('σ_hh: %.1f meV' % (sigma_hh * 1e3))
    print('ɛ_hh: %.0f, %.0f, %.0f, %.0f meV' % tuple(peak_hh * 1e3))
    print('\n', flush=True)

    data_hh = array([
        exciton_vo_nomb_vec(
            E_vec - peak_hh[ii],
            gamma_hh,
            sigma_hh,
            nx,
            ny,
            sys_hh,
        ) for nx, ny in states_hh_vec[ii]
    ]) * exp(-sys_hh.beta * E_vec)

    data_lh = array([
        exciton_vo_nomb_vec(
            E_vec - (popt_abs[6 * N_samples + ii] + popt_abs[4 * N_samples + ii]),
            popt_abs[7 * N_samples],
            popt_abs[7 * N_samples + 3],
            nx,
            ny,
            sys_lh,
        ) for nx, ny in states_lh_vec[ii]
    ]) * exp(-sys_hh.beta * E_vec) * popt_abs[ii]

    sum_data_hh = sum(data_hh, axis=0)
    sum_data_lh = sum(data_lh, axis=0)

    sum_data = sum_data_hh + sum_data_lh

    sum_amax = amax(sum_data)

    data_hh /= sum_amax
    data_lh /= sum_amax
    sum_data /= sum_amax

    sum_data_interp = interp1d(E_vec, sum_data)
    sum_data_at_fit = sum_data_interp(loaded_data[ii][:, 0])

    savetxt(
        'extra/extcharge/export_PL/%s_vo_hh.csv' % labels_vec[ii],
        hstack((
            E_vec.T.reshape(-1, 1),
            data_hh.T,
            sum_data.T.reshape(-1, 1),
        )),
        delimiter=',',
        newline='\n',
        header='E_vec (eV),%s,PL (sum)' %
        ','.join(['%d_%d' % (nx, ny) for nx, ny in states_hh_vec[ii]]),
    )

    savetxt(
        'extra/extcharge/export_PL/%s_vo_lh.csv' % labels_vec[ii],
        hstack((
            E_vec.T.reshape(-1, 1),
            data_lh.T,
            sum_data.T.reshape(-1, 1),
        )),
        delimiter=',',
        newline='\n',
        header='E_vec (eV),%s,PL (sum)' %
        ','.join(['%d_%d' % (nx, ny) for nx, ny in states_lh_vec[ii]]),
    )

    savetxt(
        'extra/extcharge/export_PL/%s_exp.csv' % labels_vec[ii],
        hstack((loaded_data[ii][:, 0].T.reshape(-1, 1),
                loaded_data[ii][:, 1].T.reshape(-1, 1))),
        delimiter=',',
        newline='\n',
        header='E_vec (eV),PL (sum)',
    )

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, len(states_hh_vec[ii]) + len(states_hh_vec[ii]) + 2)
    ]

    for jj, (data_state, state) in enumerate(zip(data_hh, states_hh_vec[ii])):
        ax[ii].plot(
            E_vec,
            data_state,
            color=colors[jj + 1],
            linewidth=0.9,
            linestyle='--',
        )

    for jj, (data_state, state) in enumerate(zip(data_lh, states_lh_vec[ii])):
        ax[ii].plot(
            E_vec,
            data_state,
            color=colors[jj - 2],
            linewidth=0.9,
            linestyle='--',
        )

    ax[ii].plot(
        E_vec,
        sum_data_hh,
        color=colors[0],
        linewidth=0.9,
    )

    ax[ii].plot(
        E_vec,
        sum_data_lh,
        color=colors[-1],
        linewidth=0.9,
    )

    ax[ii].plot(
        E_vec,
        sum_data,
        color='k',
        linewidth=1.8,
        label='AIC: $%.3f$\nAdj $R^2$: $%.3f$' % (
            aic_criterion(loaded_data[ii][:, 1], sum_data_at_fit, len(popt)),
            adj_r_squared(loaded_data[ii][:, 1], sum_data_at_fit, len(popt)),
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
    ax[ii].set_ylim(0, 1.1)

    lg = ax[ii].legend(
        loc='upper left',
        title=(r'%s: $%.1f \times %.1f$ nm' %
               (labels_vec[ii], sys_hh.size_Lx, sys_hh.size_Ly)) +
        '\n' + (r'$\epsilon_{hh}$: $%.1f\pm%.1f$ meV' %
                (peak_hh[ii] * 1e3,
                 sqrt(extra_dict_params['pcov'][ii + 2][ii + 2]) * 1e3)) +
        '\n' + (r'$\Gamma$: $%.1f\pm%.1f$ meV' %
                (gamma * 1e3, sqrt(extra_dict_params['pcov'][0][0]) * 1e3)) +
        '\n' + (r'$\sigma$: $%.1f\pm%.1f$ meV' %
                (sigma * 1e3, sqrt(extra_dict_params['pcov'][1][1]) * 1e3)),
        prop={'size': 12},
    )
    lg.get_title().set_fontsize(14)


savetxt(
    'extra/extcharge/export_PL/vo_popt.csv',
    array(popt).reshape((1, -1)),
    delimiter=',',
    newline='\n',
    header='gamma_hh (eV),sigma_hh (eV),%s' %
    ', '.join(['peak_hh (%s) (eV)' % l for l in labels_vec]),
)

savetxt(
    'extra/extcharge/export_PL/vo_pcov.csv',
    array(extra_dict_params['pcov']),
    delimiter=',',
    newline='\n',
    header='gamma_hh (eV),sigma_hh (eV),%s' %
    ', '.join(['peak_hh (%s) (eV)' % l for l in labels_vec]),
)

print(popt)

ax[0].set_xlabel(r'$\epsilon$ (eV)')

for ii, file_id in enumerate(labels_vec):
    plot_PL(ii, sys_hh, sys_lh, popt, extra_dict_params)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'cm_be_polar_vo',
    transparent=True,
)

plt.show()
