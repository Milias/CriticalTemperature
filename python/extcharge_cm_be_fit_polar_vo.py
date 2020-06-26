from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([3 * 6.8, 5.3]))

plt.ion()
n_x, n_y = 4, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.52, 0.45, 294  # K

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

N_samples = len(labels_vec)

lines_exp_list = [None] * N_samples
lines_data_list = [None] * N_samples

sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)

max_state = 8
states_hh_vec = [states_sorted_os(max_state, Lx, Ly) for Lx, Ly in sizes_vec]
states_lh_vec = [states_sorted_os(max_state, Lx, Ly) for Lx, Ly in sizes_vec]


def cou_energy(sys):
    return -sys.c_aEM * sys.c_hbarc / sqrt(sys.size_Lx**2 +
                                           sys.size_Ly**2) / sys.eps_mat


def model_PL(calc_func, loaded_data, size_vec, hwhm_vec, sys, lines_data_list):
    def model_PL_func(xdata, *popt):
        gamma_hh, sigma_hh = popt[0], popt[1]
        peak_hh = array(popt[2:6])

        print(time.strftime('%X'))
        print('Γ_hh: %.2f meV' % (gamma_hh * 1e3))
        print('σ_hh: %.1f meV' % (sigma_hh * 1e3))
        print('ɛ_hh: %.0f, %.0f, %.0f, %.0f meV' % tuple(peak_hh * 1e3))
        print('\n', flush=True)

        sum_model_all = []

        for ii in range(N_samples):
            sys.size_Lx, sys.size_Ly = size_vec[ii]
            sys.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            sum_model = sum(
                array([
                    calc_func(
                        xdata_ii - peak_hh[ii],
                        gamma_hh,
                        sigma_hh,
                        nx,
                        ny,
                        sys,
                    ) for nx, ny in states_hh_vec[ii]
                ]),
                axis=0,
            )

            sum_model /= amax(sum_model)

            lines_data_list[ii].set_ydata(sum_model)
            fig.canvas.draw()
            fig.canvas.flush_events()

            sum_model_all.extend(sum_model)

        return array(sum_model_all)

    return model_PL_func


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


loaded_data = array([
    load_raw_PL_data('extra/data/extcharge/PL-%s.txt' % label, 0.15)
    for label in labels_vec
])

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]
concat_data = concatenate(loaded_data[:])

p0_values = (
    30e-3,
    10e-3,
    *tuple(peak_eV_vec),
)
lower_bounds = (
    0,
    0,
    0,
    0,
    0,
    0,
)

for ii in range(N_samples):
    sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
    sys_hh.set_hwhm(*hwhm_vec[ii])

    xdata_ii = loaded_data[ii][:, 0]

    lines_data_list[ii], = ax[ii].plot(
        xdata_ii,
        zeros_like(xdata_ii),
        color='k',
        linewidth=1.8,
    )

    lines_exp_list[ii], = ax[ii].plot(
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

    ax[ii].set_xlim(xdata_ii[0], xdata_ii[-1])
    ax[ii].set_ylim(0, 1.1)

    lg = ax[ii].legend(
        loc='upper left',
        title=(r'%s: $%.1f \times %.1f$ nm' % (
            labels_vec[ii],
            sys_hh.size_Lx,
            sys_hh.size_Ly,
        )),
        prop={'size': 12},
    )
    lg.get_title().set_fontsize(14)

    ax[0].set_xlabel(r'$\epsilon$ (eV)')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.canvas.draw()
    fig.canvas.flush_events()

popt, pcov = time_func(
    curve_fit,
    model_PL(
        exciton_vo_vec,
        loaded_data,
        sizes_vec,
        hwhm_vec,
        sys_hh,
        lines_data_list,
    ),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=p0_values,
    bounds=(lower_bounds, inf),
    method='trf',
)

print(popt.tolist(), flush=True)

file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

save_data(
    'extra/extcharge/cm_be_polar_fit_params_vo_%s' % file_id,
    [popt],
    extra_data={
        'labels_vec': labels_vec,
        'sizes_vec': sizes_vec,
        'hwhm_vec': hwhm_vec,
        'states_hh_vec': states_hh_vec,
        'states_lh_vec': states_lh_vec,
        'pcov': pcov.tolist(),
    },
)
