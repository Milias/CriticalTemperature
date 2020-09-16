from common import *


def load_popt(popt, global_dict, var_list):
    for p, var in zip(popt, var_list):
        global_dict[var] = p


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


plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

file_id_params_PL = 'rBYyUhVlQ327WuiwjHkVhQ'

extra_dict_params_PL = {}
popt_PL = load_data(
    'extra/extcharge/cm_be_polar_fit_params_vo_%s' % file_id_params_PL,
    extra_dict_params_PL,
)

file_id_params_abs = '0NCcd7uWQOqeUUptm6rk9A'

extra_dict_params_abs = {}
popt_abs = load_data(
    'extra/extcharge/cm_be_polar_fit_params_abs_vo_%s' % file_id_params_abs,
    extra_dict_params_abs,
)

labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]

N_samples = len(labels_vec)
E_max_data = 0.15

loaded_data = [
    array(
        load_raw_PL_data('extra/data/extcharge/PL-%s.txt' % label, E_max_data))
    for label in labels_vec
]

E_min_abs_data, E_max_abs_data = 0.1, 0.65

loaded_abs_data = array([
    load_raw_Abs_data(
        'extra/data/extcharge/Abs_%s.txt' % label,
        E_min_abs_data,
        E_max_abs_data,
    ) for label in labels_vec
])

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]

max_state = 8


def compute_PL(E_vec, popt, ii, sys, extra_dict):
    globals().update(extra_dict)
    N_samples = 4

    sys_hh = system_data(
        sys.params.m_e,
        sys.params.m_hh,
        sys.params.eps_sol,
        sys.params.T,
        sys.params.size_d,
        0,
        0,
        0,
        0,
        sys.params.eps_sol,
    )
    sys_lh = system_data(
        sys.params.m_e,
        sys.params.m_lh,
        sys.params.eps_sol,
        sys.params.T,
        sys.params.size_d,
        0,
        0,
        0,
        0,
        sys.params.eps_sol,
    )

    states_vec = states_sorted_os(
        max_state,
        sizes_vec[ii][0],
        sizes_vec[ii][1],
    )

    gamma, sigma = popt[0], popt[1]
    peak_hh = array(popt[2:6])

    E_vec_local = E_vec + peak_eV_vec[ii]
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
        ) for nx, ny in states_vec
    ]) * exp(-sys.d_params.beta * E_vec)

    data_lh = array([
        exciton_vo_nomb_vec(
            E_vec -
            (popt_abs[6 * N_samples + ii] + popt_abs[4 * N_samples + ii]),
            popt_abs[7 * N_samples],
            popt_abs[7 * N_samples + 3],
            nx,
            ny,
            sys_lh,
        ) for nx, ny in states_vec
    ]) * exp(-sys.d_params.beta * E_vec) * popt_abs[ii]

    sum_data_hh = sum(data_hh, axis=0)
    sum_data_lh = sum(data_lh, axis=0)

    sum_data = sum_data_hh + sum_data_lh

    sum_amax = amax(sum_data)

    data_hh /= sum_amax
    data_lh /= sum_amax
    sum_data /= sum_amax

    return sum_data, data_hh, data_lh


def srt_dist(t, E, tau, f_dist_eq, f_dist_zero, params_eq, params_zero):
    if t < 0:
        return f_dist_zero(E, *params_zero)

    else:
        return f_dist_eq(E, *params_eq) * (1 - exp(-t / tau)) + f_dist_zero(
            E, *params_zero) * exp(-t / tau)


def f_dist_eq(E, beta, E_max):
    return exp(-beta * (E - E_max))


def f_dist_zero(E, mu, sigma):
    return stats.norm.pdf(E, loc=mu, scale=sigma)


fit_vars_list = [
    'fdepl',
    'fse',
    'tau',
]


def model(ta_fit_abs_data, E_vec, srt_time):
    E_max = ta_fit_abs_data[ta_fit_abs_data[:, -1].argmax(), 0]

    print(
        'time: %.3f ps, E_max: %.3f eV\n' % (srt_time, E_max),
        flush=True,
    )

    ta_fit_abs_interp = interp1d(
        ta_fit_abs_data[:, 0],
        ta_fit_abs_data[:, -1],
    )
    abs_data = ta_fit_abs_interp(E_vec)
    """
    ta_fit_PL_data, _, _ = compute_PL(
        E_vec,
        popt_PL,
        2,
        sys,
        extra_dict_params_PL,
    )
    """
    def model_fun(xdata, *popt, return_all=False):
        load_popt(popt, globals(), fit_vars_list)

        dist_data = srt_dist(
            srt_time,
            E_vec,
            tau,
            f_dist_eq,
            f_dist_zero,
            (sys.d_params.beta, E_max),
            (E_max, 0.033179 / (2 * sqrt(2 * log(2)))),
        )

        if return_all:
            return (
                abs_data * (1 - fdepl - fse * dist_data),
                abs_data * fdepl,
                #abs_data * fse * dist_data,
                fse * dist_data,
            )
        else:
            return abs_data * (1 - fdepl - fse * dist_data)

    return model_fun


#To HH: 2.4188 eV (512.646 nm), to LH: 480 nm (1240.0/480 eV), to Cont: 400 nm (1240/400 eV)
#Sigma = 33.179 meV

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

ta_data_list = []

n_files = 259

for i in range(n_files):
    with open('extra/data/ta_analysis/HH/BS066_HHEx_1.0mW_%d.txt' % i) as f:
        ta_data_list.append(loadtxt(f))

ta_data = array(ta_data_list)
ta_data = ta_data[:, ::-1, :]

with open('extra/data/ta_analysis/HH/times_BS066_HHEx_1.0mW.txt') as f:
    ta_times = loadtxt(f)[:, 1]

with open('extra/extcharge/export_PL/BS066_vo_hh.csv') as f:
    ta_fit_PL_data = loadtxt(f, delimiter=',')

with open('extra/extcharge/export_abs/BS066_vo_sum.csv') as f:
    ta_fit_abs_data = loadtxt(f, delimiter=',')

p0_values = (0.1, 0.1, 1.0)
bounds = ((0, 0, 0), (1, 1, float('inf')))


def full_fit_at_time_idx(time_idx, ta_data, ta_fit_abs_data):
    time_value = ta_times[time_idx]

    eV_min = max(ta_data[time_idx, 0, 0], ta_fit_abs_data[0, 0])
    eV_max = min(ta_data[time_idx, -1, 0], ta_fit_abs_data[-1, 0])

    print('Range: [%.4f, %.4f] eV' % (eV_min, eV_max))

    E_vec = linspace(eV_min, eV_max, 1 << 8)

    ta_data_mask = (ta_data[time_idx, :, 0] > eV_min) * (ta_data[time_idx, :,
                                                                 0] < eV_max)

    model_func = model(
        #ta_fit_abs_data,
        loaded_abs_data[2],
        ta_data[time_idx, ta_data_mask, 0],
        time_value,
    )

    popt, pcov = time_func(
        curve_fit,
        model_func,
        ta_data[time_idx, ta_data_mask, 0],
        ta_data[time_idx, ta_data_mask, 1],
        p0=p0_values,
        bounds=bounds,
        method='trf',
    )

    load_popt(popt, globals(), fit_vars_list)

    print(time.strftime('%X'))
    print(', '.join([
        '%s: %s %s' % (v, '%.4f', u)
        for v, u in zip(fit_vars_list, ['', '', 'ps'])
    ]) % tuple(popt))
    print('\n', flush=True)

    return E_vec, popt, pcov


time_idx = -1
time_value = ta_times[time_idx]
E_vec, popt, pcov = full_fit_at_time_idx(
    time_idx,
    ta_data,
    loaded_abs_data[2],
)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, n_files)
]

vis_fit, vis_depl, vis_se = model(
    #ta_fit_abs_data,
    loaded_abs_data[2],
    E_vec,
    time_value,
)(
    E_vec,
    *popt,
    return_all=True,
)

ax[0].plot(
    E_vec,
    vis_fit,
    color='k',
    label='Transient fit',
)

ax[0].plot(
    E_vec,
    vis_depl,
    color='m',
    label='Depletion',
)

ax[0].plot(
    E_vec,
    vis_se,
    color='c',
    label='Stimulated Emission',
)

ax[0].plot(
    #ta_fit_abs_data[:, 0],
    #ta_fit_abs_data[:, -1],
    loaded_abs_data[2][:, 0],
    loaded_abs_data[2][:, -1],
    color='k',
    linestyle='--',
    linewidth=0.9,
    label='Steady-state fit',
)

ax[0].plot(
    ta_data[time_idx, :, 0],
    ta_data[time_idx, :, 1],
    colors[time_idx],
    linewidth=0.8,
    label='%.2f ps' % time_value,
)

ax[0].set_xlim(E_vec[0], E_vec[-1])
ax[0].set_ylim(0, 1.1)

ax[0].legend(
    loc=0,
    prop={'size': 12},
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TA_Analysis/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()
