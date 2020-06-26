from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})


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


file_id_PL = 'rBYyUhVlQ327WuiwjHkVhQ'
params_PL_dict = {}
popt_PL = load_data(
    'extra/extcharge/cm_be_polar_fit_params_vo_%s' % file_id_PL,
    params_PL_dict,
)

globals().update(params_PL_dict)
"""
chosen_index = 0
labels_vec = [labels_vec[chosen_index]]
sizes_vec = [sizes_vec[chosen_index]]
hwhnm_vec = [hwhm_vec[chosen_index]]
"""

N_samples = len(sizes_vec)

fig_size = tuple(array([3 * 6.8, 5.3]))

plt.ion()
n_x, n_y = N_samples, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

lines_exp_list = [None] * N_samples
lines_data_list = [None] * N_samples * 5

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.52, 0.45, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)

max_state = 8

states_hh_vec = [states_sorted_os(max_state, Lx, Ly) for Lx, Ly in sizes_vec]
states_lh_vec = [states_sorted_os(max_state, Lx, Ly) for Lx, Ly in sizes_vec]

E_min_data, E_max_data = 0.1, 0.65

loaded_data = array([
    load_raw_Abs_data(
        'extra/data/extcharge/Abs_%s.txt' % label,
        E_min_data,
        E_max_data,
    ) for label in labels_vec
])

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]

concat_data = concatenate(loaded_data[:])


def cou_energy(sys):
    return -sys.c_aEM * sys.c_hbarc / sqrt(sys.size_Lx**2 +
                                           sys.size_Ly**2) / sys.eps_mat


def cont_func(energy, gamma_c, sigma, sys):
    return array(exciton_cont_vec(
        energy,
        gamma_c,
        #sigma,
        sys,
    ))


def model_abs(
    loaded_data,
    sizes_vec,
    hwhm_vec,
    sys_hh,
    sys_lh,
    lines_data_list,
):
    def model_abs_func(xdata, *popt):
        mag_peak_lh_vec = popt[:N_samples]
        mag_cont_lh_vec = popt[N_samples:2 * N_samples]
        mag_cont_hh_vec = popt[2 * N_samples:3 * N_samples]
        E_hh, E_lh = array(popt[3 * N_samples:4 * N_samples]), array(
            popt[4 * N_samples:5 * N_samples])
        energy_c_hh, energy_c_lh = array(
            popt[5 * N_samples:6 * N_samples]), array(popt[6 * N_samples:7 *
                                                           N_samples])
        gamma_hh, gamma_lh, gamma_c_hh, gamma_c_lh = popt_PL[
            0], *popt[7 * N_samples:7 * N_samples + 3]
        sigma_hh, sigma_lh = popt_PL[1], (popt[7 * N_samples +
                                               3] if len(popt) == 32 else 0)

        print(time.strftime('%X'))

        print('Γ_hh: %.1f meV, Γ_lh: %.1f meV' % (
            gamma_hh * 1e3,
            gamma_lh * 1e3,
        ))
        print('σ_hh: %.1f meV, σ_lh: %.1f meV' % (
            sigma_hh * 1e3,
            sigma_lh * 1e3,
        ))
        print('E_hh: %s meV' % ', '.join(['%.1f' % e for e in (E_hh * 1e3)]))
        print('E_lh: %s meV' % ', '.join(['%.1f' % e for e in (E_lh * 1e3)]))
        print('Γ_c_hh: %.0f meV, ɛ_c_hh: %s meV' % (
            gamma_c_hh * 1e3,
            ', '.join(['%.0f' % e for e in (energy_c_hh * 1e3)]),
        ))
        print('Γ_c_lh: %.0f meV, ɛ_c_lh: %s meV' % (
            gamma_c_lh * 1e3,
            ', '.join(['%.0f' % e for e in (energy_c_lh * 1e3)]),
        ))
        print('mag_peak_lh: %s' %
              ', '.join(['%.3f' % e for e in mag_peak_lh_vec]))
        print('mag_cont_hh: %s' %
              ', '.join(['%.3f' % e for e in mag_cont_hh_vec]))
        print('mag_cont_lh: %s' %
              ', '.join(['%.3f' % e for e in mag_cont_lh_vec]))
        print('\n', flush=True)

        sum_model_all = []

        for ii in range(N_samples):
            sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
            sys_hh.set_hwhm(*hwhm_vec[ii])

            sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
            sys_lh.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            data_cont_hh = cont_func(
                xdata_ii - energy_c_hh[ii],
                gamma_c_hh,
                sigma_hh,
                sys_hh,
            ) * mag_cont_hh_vec[ii]

            data_cont_lh = cont_func(
                xdata_ii - energy_c_lh[ii],
                gamma_c_lh,
                sigma_lh,
                sys_lh,
            ) * mag_cont_lh_vec[ii]

            data_hh_sum = sum(
                array([
                    exciton_vo_nomb_vec(
                        xdata_ii - (energy_c_hh[ii] + E_hh[ii]),
                        gamma_hh,
                        sigma_hh,
                        nx,
                        ny,
                        sys_hh,
                    ) for nx, ny in states_hh_vec[ii]
                ]),
                axis=0,
            )

            data_lh_sum = sum(
                array([
                    exciton_vo_nomb_vec(
                        xdata_ii - (energy_c_lh[ii] + E_lh[ii]),
                        gamma_lh,
                        sigma_lh,
                        nx,
                        ny,
                        sys_lh,
                    ) for nx, ny in states_lh_vec[ii]
                ]),
                axis=0,
            )

            data_hh_sum /= amax(data_hh_sum)
            data_lh_sum /= amax(data_lh_sum) / mag_peak_lh_vec[ii]

            sum_model = data_hh_sum + data_lh_sum + data_cont_hh + data_cont_lh

            sum_model_max = amax(sum_model)

            data_hh_sum /= sum_model_max
            data_lh_sum /= sum_model_max
            data_cont_hh /= sum_model_max
            data_cont_lh /= sum_model_max
            sum_model /= sum_model_max

            lines_data_list[0 * N_samples + ii].set_ydata(data_hh_sum)
            lines_data_list[1 * N_samples + ii].set_ydata(data_lh_sum)
            lines_data_list[2 * N_samples + ii].set_ydata(data_cont_hh)
            lines_data_list[3 * N_samples + ii].set_ydata(data_cont_lh)
            lines_data_list[4 * N_samples + ii].set_ydata(sum_model)

            fig.canvas.draw()
            fig.canvas.flush_events()

            sum_model_all.extend(sum_model)

        return array(sum_model_all)

    return model_abs_func


p0_values = [
    *tuple([0.65] * N_samples),
    #
    *tuple([0.2] * N_samples),
    #
    *tuple([0.42] * N_samples),
    #
    *tuple([-190e-3] * N_samples),
    #
    *tuple([-270e-3] * N_samples),
    #
    *tuple([2600e-3] * N_samples),
    #
    *tuple([2830e-3] * N_samples),
    #
    20e-3,
    100e-3,
    100e-3,
    #
    40e-3,
]

p0_values = [
    0.4689286523283527,
    0.4705934867318183,
    0.42660838449347993,
    0.49422783331004155,
    0.14964880780215126,
    0.16009417835439027,
    0.18083019351660637,
    0.20598313894527162,
    0.5144168286369252,
    0.4592199751407625,
    0.42586480687922074,
    0.4710742111020432,
    -0.15000000000000002,
    -0.15000000009059195,
    -0.15036611632620348,
    -0.1500000000000014,
    -0.2994773202693713,
    -0.29967880900698385,
    -0.28729345712174187,
    -0.2914931569789018,
    2.5640961690385544,
    2.560010064485083,
    2.5629001122116666,
    2.5626990010235433,
    2.8392992828746797,
    2.8397834792367385,
    2.831258587797356,
    2.838906921351495,
    0.005000000000000001,
    0.13796239708113084,
    0.08629806214022008,
    0.05293669235348955,
]

p0_values = [
    0.4107834061791068,
    0.42952163596209086,
    0.38722611854155825,
    0.4453141089274792,
    0.17535766368553451,
    0.1844476512267471,
    0.19912588070019704,
    0.22743184149300624,
    0.47959785225474344,
    0.4269666286986492,
    0.4006079690090578,
    0.44119909387262823,
    -0.1322727880667906,
    -0.1356140078803231,
    -0.1366174235069333,
    -0.13517388325297072,
    -0.30630616841193214,
    -0.3047608470925833,
    -0.2952629392877876,
    -0.29749779857083447,
    2.5462585027761144,
    2.5454983189864357,
    2.5490208126987133,
    2.5478095606555007,
    2.8383591322814063,
    2.8384926640766714,
    2.831832425220786,
    2.8388810448833817,
    0.005000000000000001,
    0.054141177036319516,
    0.09872438817100768,
    0.05315713125111064,
]

min_xdata = tuple([l[0, 0] for l in loaded_data])
max_xdata = tuple([l[-1, 0] for l in loaded_data])

print(min_xdata)
print(max_xdata)

lower_bounds = (
    *tuple([0] * N_samples * 3),
    #
    *tuple([-inf] * N_samples),
    #
    *tuple([-inf] * N_samples),
    #
    *min_xdata,
    #
    *min_xdata,
    #
    5e-3,
    5e-3,
    5e-3,
    #
    5e-3,
)

upper_bounds = (
    *tuple([1] * N_samples * 3),
    #
    *tuple([-100e-3] * N_samples),
    #
    *tuple([-250e-3] * N_samples),
    #
    *max_xdata,
    #
    *max_xdata,
    #
    150e-3,
    150e-3,
    150e-3,
    #
    100e-3,
)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 2)
]

for ii in range(N_samples):
    sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
    sys_hh.set_hwhm(*hwhm_vec[ii])

    xdata_ii = loaded_data[ii][:, 0]

    lines_data_list[0 * N_samples + ii], = ax[ii].plot(
        xdata_ii,
        zeros_like(xdata_ii),
        color=colors[0],
        linewidth=0.9,
    )

    lines_data_list[1 * N_samples + ii], = ax[ii].plot(
        xdata_ii,
        zeros_like(xdata_ii),
        color=colors[-1],
        linewidth=0.9,
    )

    lines_data_list[2 * N_samples + ii], = ax[ii].plot(
        xdata_ii,
        zeros_like(xdata_ii),
        color='m',
        linewidth=0.9,
    )

    lines_data_list[3 * N_samples + ii], = ax[ii].plot(
        xdata_ii,
        zeros_like(xdata_ii),
        color='g',
        linewidth=0.9,
    )

    lines_data_list[4 * N_samples + ii], = ax[ii].plot(
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
        loc='upper right',
        title=(r'%s: $%.1f \times %.1f$ nm' % (
            labels_vec[ii],
            sys_hh.size_Lx,
            sys_hh.size_Ly,
        )),
        prop={'size': 11},
    )
    lg.get_title().set_fontsize(12)

    ax[0].set_xlabel(r'$\epsilon$ (eV)')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.canvas.draw()
    fig.canvas.flush_events()

popt, pcov = time_func(
    curve_fit,
    model_abs(
        loaded_data,
        sizes_vec,
        hwhm_vec,
        sys_hh,
        sys_lh,
        lines_data_list,
    ),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=p0_values,
    bounds=(lower_bounds, upper_bounds),
    method='trf',
)

print(popt.tolist(), flush=True)

file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

save_data(
    'extra/extcharge/cm_be_polar_fit_params_abs_vo_%s' % file_id,
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
