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

loaded_abs_data = [
    array(
        load_raw_Abs_data(
            'extra/data/extcharge/Abs_%s.txt' % label,
            E_min_abs_data,
            E_max_abs_data,
        )) for label in labels_vec
]


def srt_dist(t, E, alpha, f_dist_eq, f_dist_zero, params_eq, params_zero):
    return f_dist_eq(E, *params_eq) * (1 - alpha) + f_dist_zero(
        E, *params_zero) * alpha


def f_dist_eq(E, beta):
    return beta * exp(-beta * E)


def f_dist_zero(E, mu, sigma):
    return stats.norm.pdf(E, loc=mu, scale=sigma)


with open('config/ta_srt_approx.yaml') as f:
    print('Loading "%s".' % f.name)
    ta_srt_dict = yaml.load(f, Loader=yaml.CLoader)

fit_vars_list = list(ta_srt_dict['fit_vars'].keys())

fit_units_list = list(
    [ta_srt_dict['fit_vars'][var]['unit'] for var in ta_srt_dict['fit_vars']])

p0_values = tuple(
    [ta_srt_dict['fit_vars'][var]['p0'] for var in ta_srt_dict['fit_vars']])

bounds = array([
    tuple(ta_srt_dict['fit_vars'][var]['bounds'])
    for var in ta_srt_dict['fit_vars']
]).T


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

    def model_fun(xdata, *popt, return_all=False):
        load_popt(popt, globals(), fit_vars_list)

        dist_data = srt_dist(
            srt_time,
            E_vec,
            alpha,
            f_dist_eq,
            f_dist_zero,
            (sys.d_params.beta, ),
            #(2.4188, 0.033179 / (2 * sqrt(2 * log(2)))),
            (Epump, 0.033179 / (2 * sqrt(2 * log(2)))),
        )

        hhhh = hhhhMag * stats.norm.pdf(
            E_vec,
            loc=hhhhLoc,
            scale=hhhhSig,
        )

        hhlh = hhlhMag * stats.norm.pdf(
            E_vec,
            loc=hhlhLoc,
            scale=hhlhSig,
        )

        result = abs_data * (1 - fdepl - fse * dist_data) + hhhh + hhlh

        if return_all:
            return {
                'all': result,
                'depl': abs_data * fdepl,
                'se': abs_data * fse * dist_data,
                'abs': abs_data,
                'hhhh': hhhh,
                'hhlh': hhlh,
            }
        else:
            return result

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


def full_fit_at_time_idx(time_idx, ta_data, ta_fit_abs_data):
    time_value = ta_times[time_idx]

    eV_min = max(ta_data[time_idx, 0, 0], ta_fit_abs_data[0, 0])
    eV_max = min(ta_data[time_idx, -1, 0], ta_fit_abs_data[-1, 0])

    print('Range: [%.4f, %.4f] eV' % (eV_min, eV_max))

    E_vec = linspace(eV_min, eV_max, 1 << 8)

    ta_data_mask = (ta_data[time_idx, :, 0] > eV_min) * (ta_data[time_idx, :,
                                                                 0] < eV_max)

    model_func = model(
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

    print(time.strftime('%X'))
    print(', '.join([
        '%s: %s %s' % (v, '%.4f', u)
        for v, u in zip(fit_vars_list, fit_units_list)
    ]) % tuple(popt))
    print('\n', flush=True)

    return E_vec, popt, pcov


colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, n_files)
]


def plot_fit(time_idx):
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': True,
    })

    n_x, n_y = 1, 1
    fig = plt.figure(figsize=fig_size)
    ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

    time_value = ta_times[time_idx]
    E_vec, popt, pcov = full_fit_at_time_idx(
        time_idx,
        ta_data,
        loaded_abs_data[2],
    )

    vis_data_dict = model(
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
        vis_data_dict['all'],
        color='k',
        label='Transient fit',
    )

    ax[0].plot(
        E_vec,
        vis_data_dict['depl'],
        color='m',
        label='Depletion',
    )

    ax[0].plot(
        E_vec,
        vis_data_dict['se'],
        color='c',
        label='Stimulated Emission',
    )

    ax[0].plot(
        E_vec,
        vis_data_dict['abs'],
        color='k',
        linestyle='--',
        linewidth=0.9,
        label='Steady-state',
    )

    ax[0].plot(
        E_vec,
        vis_data_dict['hhhh'],
        color='r',
        linestyle='--',
        linewidth=0.9,
        label='$hhhh$',
    )

    ax[0].plot(
        E_vec,
        vis_data_dict['hhlh'],
        color='b',
        linestyle='--',
        linewidth=0.9,
        label='$hhlh$',
    )

    ax[0].plot(
        ta_data[time_idx, :, 0],
        ta_data[time_idx, :, 1],
        colors[time_idx],
        linewidth=0.8,
        label='%.2f ps' % time_value,
    )

    ax[0].text(
        2.325,
        0.93,
        '\n'.join([
            r'depl: $%.3f$' % popt[0],
            r'se: $%.3f$' % popt[1],
            r'$\alpha$: $%.3f$' % popt[2],
            r'Epump: $%.3f$ eV' % popt[3],
            r'hhhh: ($%.3f$, $%.3f$ eV, $%.3f$ eV)' % tuple(popt[4::2]),
            r'hhlh: ($%.3f$, $%.3f$ eV, $%.3f$ eV)' % tuple(popt[5::2]),
        ]),
        fontsize=7,
    )

    ax[0].set_xlim(E_vec[0], E_vec[-1])
    ax[0].set_ylim(0, 1.1)

    ax[0].set_xlabel('$E$ (eV)')

    ax[0].legend(
        loc='upper right',
        prop={'size': 8},
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(
        '/storage/Reference/Work/University/PhD/TA_Analysis/plots_all/%s_%03d_%s.png'
        % (os.path.splitext(os.path.basename(__file__))[0], time_idx, 'v1'),
        #transparent=True,
        dpi=300,
    )

    plt.close()


pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
pool.map(plot_fit, range(ta_times.size))
