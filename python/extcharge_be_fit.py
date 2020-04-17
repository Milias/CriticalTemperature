from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
    'errorbar.capsize': 5,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.45, 0.52, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)


def cou_energy(LS, area, sys):
    #return -(sys.c_hbarc * pi)**2 * 0.5 / sys.m_p * LS**2 / (
    return -(sys.c_hbarc * pi)**2 * 0.5 / (sys.m_e + sys.m_h) * LS**2 / (
        area)**2 + sys.c_aEM * sys.c_hbarc / sys.eps_mat / LS


def plot_Abs(ii, LS_samples_vec, sys_hh, sys_lh, file_id_params, marker,
             fit_label):
    extra_dict = {}
    popt = load_data(
        'extra/extcharge/cm_be_polar_fit_params_abs_%s' % file_id_params,
        extra_dict,
    )
    globals().update(extra_dict)

    sizes_vec = array(extra_dict['sizes_vec'])

    LS_samples_vec = sqrt(sizes_vec[:, 0]**2 + sizes_vec[:, 1]**2)
    area_vec = product(sizes_vec, axis=1)

    colors = [
        matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8])
        for h in linspace(0, 0.7, LS_samples_vec.size)
    ]

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

        perr = zeros((23, ))
        perr[:11] = sqrt(diag(pcov))[:11]
        perr[11:15] = sqrt(diag(pcov))[11]
        perr[15:] = sqrt(diag(pcov))[12:]

    E_hh = peak_hh_vec - energy_c
    E_hh_err = perr[2:6] + perr[11:15]
    E_lh = peak_lh_vec - energy_c
    E_lh_err = perr[6:10] + perr[11:15]

    if marker == 'o':
        popt, pcov = curve_fit(
            lambda x, E0: E0 + cou_energy(x, area_vec, sys_lh),
            LS_samples_vec,
            E_lh,
            p0=(E_lh[-1], ),
        )

        print(
            'E_lh: %.0f±%.0f meV' % (popt[0] * 1e3, sqrt(pcov[0, 0]) * 1e3),
            flush=True,
        )

        ax[0].errorbar(
            [LS_samples_vec[ii]],
            [(popt[0] + cou_energy(LS_samples_vec[ii], area_vec[ii], sys_lh)) *
             1e3],
            yerr=[sqrt(pcov[0, 0]) * 1e3],
            capsize=3.0,
            linestyle='',
            marker='x',
            markersize=6.0,
            color='k',
            zorder=100,
        )

        if ii == 0:
            ax[0].axhline(
                y=popt[0] * 1e3,
                #marker=marker,
                markerfacecolor=(1, 1, 1, 1),
                markersize=5.0,
                markeredgewidth=1.0,
                linestyle='-',
                linewidth=0.9,
                color='k',
                label=r'$%.0f\pm%.0f$ meV' % (
                    popt * 1e3,
                    sqrt(pcov[0, 0]) * 1e3,
                ),
                zorder=-100,
            )

        popt, pcov = curve_fit(
            lambda x, E0: E0 + cou_energy(x, area_vec, sys_hh),
            LS_samples_vec,
            E_hh,
            p0=(E_hh[-1], ),
        )

        print(
            'E_hh: %.0f±%.0f meV' % (popt[0] * 1e3, sqrt(pcov[0, 0]) * 1e3),
            flush=True,
        )

        ax[1].errorbar(
            [LS_samples_vec[ii]],
            [(popt[0] + cou_energy(LS_samples_vec[ii], area_vec[ii], sys_hh)) *
             1e3],
            yerr=[sqrt(pcov[0, 0]) * 1e3],
            capsize=3.0,
            linestyle='',
            marker='x',
            markersize=6.0,
            color='k',
            zorder=100,
        )

        if ii == 0:
            ax[1].axhline(
                y=popt[0] * 1e3,
                #marker=marker,
                markerfacecolor=(1, 1, 1, 1),
                markersize=5.0,
                markeredgewidth=1.0,
                linestyle='-',
                linewidth=0.9,
                color='k',
                label=r'$%.0f\pm%.0f$ meV' % (
                    popt * 1e3,
                    sqrt(pcov[0, 0]) * 1e3,
                ),
                zorder=-100,
            )

    ax[0].errorbar(
        [LS_samples_vec[ii]],
        [E_lh[ii] * 1e3],
        yerr=[E_lh_err[ii] * 1e3],
        capsize=5.0 if marker == 'o' else 2.5,
        linestyle='',
        marker=marker,
        color=(*tuple(colors[ii]), 1 if marker == 'o' else 0.5),
        markeredgecolor=(*tuple(colors[ii]), 1 if marker == 'o' else 0.5),
        markerfacecolor=(1, 1, 1, 1),
        markersize=6.0 if marker == 'o' else 3.5,
        markeredgewidth=1.2,
    )

    ax[1].errorbar(
        [LS_samples_vec[ii]],
        [E_hh[ii] * 1e3],
        yerr=[E_hh_err[ii] * 1e3],
        capsize=5.0 if marker == 'o' else 2.5,
        linestyle='',
        marker=marker,
        color=(*tuple(colors[ii]), 1 if marker == 'o' else 0.5),
        markeredgecolor=(*tuple(colors[ii]), 1 if marker == 'o' else 0.5),
        markerfacecolor=(1, 1, 1, 1),
        markersize=6.0 if marker == 'o' else 3.5,
        markeredgewidth=1.2,
    )


labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]
sizes_vec = [
    (29.32, 5.43),
    (26.11, 6.42),
    (25.4, 8.05),
    (13.74, 13.37),
]
hwhm_vec = [
    (3.3, 0.81),
    (3.34, 1.14),
    (2.9, 0.95),
    (2.17, 1.85),
]

sizes_vec = array(sizes_vec)
hwhm_vec = array(hwhm_vec)

LS_samples_vec = sqrt(sizes_vec[:, 0]**2 + sizes_vec[:, 1]**2)

file_id_list = ['3bO_Kr4XTPuW8jB2-X2X8g', 'l9dk3m1uRtqABTVesYrnNg']
marker_list = ['D', 'o']
fit_label_list = ['v', 'c']

for fid, m, fit_label in zip(file_id_list, marker_list, fit_label_list):
    for ii, file_id in enumerate(labels_vec):
        plot_Abs(ii, LS_samples_vec, sys_hh, sys_lh, fid, m, fit_label)

ax[0].set_xticks([])
ax[1].set_xlabel(r'$L_S$ (nm)')
ax[1].set_ylabel(r'$E$ (meV)')
ax[1].yaxis.set_label_coords(-0.1, 1.0)

lg = ax[0].legend(loc=0, title=r'$\langle E_{lh} \rangle$', prop={'size': 13})
lg.get_title().set_fontsize(13)

lg = ax[1].legend(loc=0, title=r'$\langle E_{hh} \rangle$', prop={'size': 13})
lg.get_title().set_fontsize(13)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    ('be_fit_abs'),
    transparent=True,
)

plt.show()
