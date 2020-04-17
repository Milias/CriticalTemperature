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


E_min_data, E_max_data = 0.1, 0.6

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

N_E = 1 << 9

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.45, 0.52, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)


def plot_Abs(ii, sys_hh, sys_lh, popt, extra_dict):
    globals().update(extra_dict)

    sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
    sys_hh.set_hwhm(*hwhm_vec[ii])

    sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
    sys_lh.set_hwhm(*hwhm_vec[ii])

    pcov = array(extra_dict['pcov'])
    perr = sqrt(diag(pcov))

    if popt.size == 23:
        # args:
        # gamma_hh, gamma_lh, peak_hh (4), peak_lh (4)
        # gamma_c, energy_c (4)
        # mag_peak_hh (4), mag_peak_lh (4), mag_cont (4)
        gamma_hh, gamma_lh = popt[:2]
        peak_hh_vec = array(popt[2:6])
        peak_lh_vec = array(popt[6:10])
        gamma_c, energy_c = popt[10], array(popt[11:15])
        mag_peak_lh_vec = popt[15:19]
        mag_cont_vec = popt[19:23]
    elif popt.size == 20:
        # args:
        # gamma_hh, gamma_lh, peak_hh (4), peak_lh (4)
        # gamma_c, energy_c
        # mag_peak_hh (4), mag_peak_lh (4), mag_cont (4)
        gamma_hh, gamma_lh = popt[:2]
        peak_hh_vec = array(popt[2:6])
        peak_lh_vec = array(popt[6:10])
        gamma_c, energy_c = popt[10], array([popt[11]] * 4)
        mag_peak_lh_vec = popt[12:16]
        mag_cont_vec = popt[16:20]

        perr = zeros((23,))
        perr[:11] = sqrt(diag(pcov))[:11]
        perr[11:15] = sqrt(diag(pcov))[11]
        perr[15:] = sqrt(diag(pcov))[12:]

    E_hh = peak_hh_vec[ii] - energy_c[ii]
    E_hh_err = perr[2 + ii] + perr[11 + ii]
    E_lh = peak_lh_vec[ii] - energy_c[ii]
    E_lh_err = perr[6 + ii] + perr[11 + ii]

    print('[%s]: E_hh - E_c = %.0f±%.0f meV' % (
        labels_vec[ii],
        E_hh * 1e3,
        E_hh_err * 1e3,
    ))

    print(
        '[%s]: E_lh - E_c = %.0f±%.0f meV' % (
            labels_vec[ii],
            E_lh * 1e3,
            E_lh_err * 1e3,
        ),
        flush=True,
    )

    E_vec = linspace(-E_min_data, E_max_data, N_E)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(
            0,
            0.7,
            (len(states_vec) + 1) * 2,
        )
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

    data_cont = lorentz_cont(E_vec - energy_c[ii], gamma_c) * mag_cont_vec[ii]

    data_hh /= amax(data_hh)
    data_lh /= amax(data_lh) / mag_peak_lh_vec[ii]

    data_hh_sum, data_lh_sum = sum(data_hh, axis=0), sum(data_lh, axis=0)

    sum_model = data_hh_sum + data_lh_sum + data_cont
    data_hh /= amax(sum_model)
    data_lh /= amax(sum_model)
    data_hh_sum /= amax(sum_model)
    data_lh_sum /= amax(sum_model)
    data_cont /= amax(sum_model)
    sum_model /= amax(sum_model)

    sum_model_interp = interp1d(E_vec, sum_model)
    sum_model_at_fit = sum_model_interp(loaded_data[ii][:, 0])

    E_vec *= 1e3
    loaded_data[ii][:, 0] *= 1e3

    for jj, (
            d_hh,
            d_lh,
            state,
    ) in enumerate(zip(
            data_hh,
            data_lh,
            states_vec,
    )):
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
        label=(r'$\epsilon_{hh}$: $%.0f\pm%.0f$ meV' % (
            peak_hh_vec[ii] * 1e3,
            perr[2 + ii] * 1e3,
        )) + '\n' + (r'$\Gamma_{hh}$: $%.0f\pm%.0f$ meV' % (
            gamma_hh * 1e3,
            perr[0] * 1e3,
        )),
    )

    ax[ii].plot(
        E_vec,
        data_lh_sum,
        color=colors[-1],
        linewidth=0.9,
        label=(r'$\epsilon_{lh}$: $%.0f\pm%.0f$ meV' % (
            peak_lh_vec[ii] * 1e3,
            perr[6 + ii] * 1e3,
        )) + '\n' + (r'$\Gamma_{lh}$: $%.0f\pm%.0f$ meV' % (
            gamma_lh * 1e3,
            perr[1] * 1e3,
        )),
    )

    ax[ii].plot(
        E_vec,
        data_cont,
        color='m',
        linewidth=0.9,
        label=(r'$\epsilon_{c}$: $%.0f\pm%.0f$ meV' % (
            energy_c[ii] * 1e3,
            perr[11 + ii] * 1e3,
        )) + '\n' + (r'$\Gamma_{c}$: $%.0f\pm%.0f$ meV' % (
            gamma_c * 1e3,
            perr[10] * 1e3,
        )),
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

    ax[ii].set_xlim(E_vec[0], E_vec[-1])
    ax[ii].set_ylim(0, None)

    ax[ii].set_xticks([0, 200, 400, 600])

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
              (E_hh * 1e3, E_hh_err * 1e3, E_lh * 1e3, E_lh_err * 1e3)),
        prop={'size': 11},
    )
    lg.get_title().set_fontsize(11)


file_id_params = '3bO_Kr4XTPuW8jB2-X2X8g'
#file_id_params = 'l9dk3m1uRtqABTVesYrnNg'

extra_dict_params = {}
popt = load_data(
    'extra/extcharge/cm_be_polar_fit_params_abs_%s' % file_id_params,
    extra_dict_params,
)

print(popt.tolist(), flush=True)

ax[0].set_xlabel(r'$\epsilon$ (meV)')

for ii, file_id in enumerate(labels_vec):
    plot_Abs(ii, sys_hh, sys_lh, popt, extra_dict_params)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('cm_be_polar_%ds_abs_%dp' % (len(states_vec), popt.size)),
    transparent=True,
)

plt.show()
