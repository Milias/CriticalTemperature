from common import *


def srt_dist(E, fse, alpha, f_eq, f_zero):
    eq_data = f_eq(E)
    zero_data = f_zero(E)

    return eq_data * fse + zero_data * alpha


def f_dist_eq(E, beta, shift):
    return exp(-beta * (E - shift))


def f_dist_zero(E, mu, sigma):
    return stats.norm.pdf(E, loc=mu, scale=sigma)


def TA_model(abs_data, ta_srt_dict, pump_case):
    var_list = list(ta_srt_dict['fit_vars_model'].keys())

    pump_sigma = ta_srt_dict['raw_data']['pump_sigma'][pump_case]

    def model_fun(xdata, *popt, return_dict=False):
        load_popt(popt, globals(), var_list)

        abs_interp = interp1d(
            abs_data[:, 0],
            abs_data[:, -1],
            bounds_error=False,
            fill_value=0.0,
        )(xdata - abs_shift)

        hh_interp = interp1d(
            abs_data[:, 0],
            abs_data[:, 1],
            bounds_error=False,
            fill_value=0.0,
        )(xdata - abs_shift)

        lh_interp = interp1d(
            abs_data[:, 0],
            abs_data[:, 2],
            bounds_error=False,
            fill_value=0.0,
        )(xdata - abs_shift)

        cont_hh_interp = interp1d(
            abs_data[:, 0],
            abs_data[:, 3],
            bounds_error=False,
            fill_value=0.0,
        )(xdata - abs_shift)

        cont_lh_interp = interp1d(
            abs_data[:, 0],
            abs_data[:, 4],
            bounds_error=False,
            fill_value=0.0,
        )(xdata - abs_shift)

        dist_data = srt_dist(
            xdata,
            fse,
            alpha,
            functools.partial(
                f_dist_eq,
                beta=sys.d_params.beta,
                shift=xdata[0],
            ),
            functools.partial(
                f_dist_zero,
                mu=pump_mu - abs_shift,
                sigma=pump_sigma,
            ),
        )

        se_hh_data = hh_interp * dist_data
        se_lh_data = lh_interp * dist_data
        depl_data = fdepl * (cont_hh_interp + cont_lh_interp)

        result = abs_interp - depl_data - se_hh_data - se_lh_data

        if return_dict:
            return {
                'all': result,
                'depl': depl_data,
                'se_hh': se_hh_data,
                'se_lh': se_lh_data,
                'abs': abs_interp,
                'hhlh': zeros_like(result),
                'hhhh': zeros_like(result),
            }
        else:
            return result

    return model_fun


def TA_fit(time_idx, ta_data, abs_data, ta_srt_dict, pump_case):
    #eV_min = max(ta_data[time_idx, 0, 0], abs_data[0, 0])
    #eV_max = min(ta_data[time_idx, -1, 0], abs_data[-1, 0])
    eV_min, eV_max = 2.35, 2.65
    E_vec = linspace(eV_min, eV_max, ta_srt_dict['settings']['N_E'])

    ta_data_mask = (ta_data[time_idx, :, 0] > eV_min) * (ta_data[time_idx, :,
                                                                 0] < eV_max)

    TA_model_func = TA_model(
        abs_data,
        ta_srt_dict,
        pump_case,
    )

    p0_values = tuple([
        ta_srt_dict['fit_vars_model'][var]['p0'] if isinstance(
            ta_srt_dict['fit_vars_model'][var]['p0'], float) else
        ta_srt_dict['fit_vars_model'][var]['p0'][pump_case]
        for var in ta_srt_dict['fit_vars_model']
    ])

    bounds = array([
        tuple(ta_srt_dict['fit_vars_model'][var]['bounds']) if isinstance(
            ta_srt_dict['fit_vars_model'][var]['bounds'], list) else tuple(
                ta_srt_dict['fit_vars_model'][var]['bounds'][pump_case])
        for var in ta_srt_dict['fit_vars_model']
    ]).T

    popt, pcov = curve_fit(
        TA_model_func,
        ta_data[time_idx, ta_data_mask, 0],
        ta_data[time_idx, ta_data_mask, 1],
        p0=p0_values,
        bounds=bounds,
        method='trf',
        maxfev=5000,
    )

    data = {
        'model': TA_model_func(ta_data[time_idx, ta_data_mask, 0], *popt),
        'data': ta_data[time_idx, ta_data_mask, 1],
        'n_params': len(popt),
    }

    return E_vec, data, popt, pcov


def plot_fit(
    time_idx,
    ta_data,
    ta_times,
    abs_data,
    ta_srt_dict,
    pump_case,
    colors,
):
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({
        'font.serif': ['Computer Modern'],
        'text.usetex': False,
    })

    fig_size = (6.8, 5.3)

    n_x, n_y = 1, 1
    fig = plt.figure(figsize=fig_size)
    ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

    time_value = ta_times[time_idx]
    E_vec, fit_data, popt, pcov = TA_fit(
        time_idx,
        ta_data,
        abs_data,
        ta_srt_dict,
        pump_case,
    )

    load_popt(popt, globals(), ta_srt_dict['fit_vars_model'].keys())

    vis_dict = TA_model(
        abs_data,
        ta_srt_dict,
        pump_case,
    )(
        E_vec,
        *popt,
        return_dict=True,
    )

    ax[0].plot(
        E_vec,
        vis_dict['hhhh'],
        color='r',
        linestyle='--',
        linewidth=0.9,
        label='$hhhh$',
    )

    ax[0].plot(
        E_vec,
        vis_dict['hhlh'],
        color='b',
        linestyle='--',
        linewidth=0.9,
        label='$hhlh$',
    )

    ax[0].plot(
        E_vec,
        vis_dict['depl'],
        color='m',
        label='Depletion',
    )

    ax[0].plot(
        ta_data[0, :, 0],
        ta_data[0, :, 1],
        color='g',
        linestyle=':',
        linewidth=0.6,
        label='Steady-state exp',
    )

    ax[0].plot(
        E_vec,
        vis_dict['se_hh'],
        color='r',
        label='Stimulated Emission (hh)',
        linewidth=0.7,
    )

    ax[0].plot(
        E_vec,
        vis_dict['se_lh'],
        color='b',
        label='Stimulated Emission (lh)',
        linewidth=0.7,
    )

    ax[0].plot(
        E_vec,
        vis_dict['abs'],
        color='k',
        linestyle='--',
        linewidth=0.9,
        label='Steady-state model',
    )

    ax[0].plot(
        E_vec,
        vis_dict['all'],
        color='k',
        label='Transient fit',
    )

    ax[0].plot(
        ta_data[time_idx, :, 0],
        ta_data[time_idx, :, 1],
        color=colors[time_idx],
        linewidth=0.8,
        label='%.2f ps' % time_value,
    )
    """
    r'hhhh: (%.3f, %.3f eV, %.1f meV)' % (
        hhhh_mag * 1e3,
        hhhh_loc,
        hhhh_sig * 1e3,
    ),
    r'hhlh: (%.3f, %.3f eV, %.1f meV)' % (
        hhlh_mag * 1e3,
        hhlh_loc,
        hhlh_sig * 1e3,
    ),
    """

    ax[0].text(
        2.3525,
        0.81,
        '\n'.join([
            r'fse: %.5f' % fse,
            r'depl: %.5f' % fdepl,
            r'alpha: %.5f' % alpha,
            r'abs_shift: %.2f meV' % (abs_shift * 1e3),
            r'pump_mu: %.4f eV' % pump_mu,
            '',
            'Adj R^2: %.5f' % adj_r_squared(**fit_data),
        ]),
        fontsize=7,
    )

    ax[0].set_xlim(E_vec[0], E_vec[-1])
    ax[0].set_ylim(0, 1.1)

    ax[0].set_xlabel('E (eV)')

    ax[0].legend(
        loc='upper right',
        prop={'size': 8},
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig_filename = '/storage/Reference/Work/University/PhD/TA_Analysis/plots_all_%s_%s/%s_%03d.png' % (
        os.path.splitext(os.path.basename(__file__))[0],
        pump_case,
        'v1',
        time_idx,
    )

    try:
        os.remove(fig_filename)
    except FileNotFoundError as e:
        pass

    fig.savefig(
        fig_filename,
        #transparent = True,
        dpi=300,
    )

    plt.close()


with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

with open('config/ta_srt_approx.yaml') as f:
    print('Loading "%s".' % f.name)
    ta_srt_dict = yaml.load(f, Loader=yaml.CLoader)

E_max_data = 0.15
E_min_abs_data, E_max_abs_data = 0.1, 0.65

abs_data = loadtxt(
    ta_srt_dict['abs_data']['folder'] +
    ta_srt_dict['abs_data']['file'] % ta_srt_dict['raw_data']['sample_label'],
    delimiter=',',
)

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    print(
        '[%d/%d] Processing "%s" ...' % (
            ii + 1,
            len(ta_srt_dict['settings']['plot_cases']),
            pump_case,
        ),
        flush=True,
    )

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(
            0,
            0.7,
            ta_srt_dict['raw_data']['n_files'][pump_case][1] -
            ta_srt_dict['raw_data']['n_files'][pump_case][0],
        )
    ]

    ta_data_list = []

    for i in range(*ta_srt_dict['raw_data']['n_files'][pump_case]):
        with open(ta_srt_dict['raw_data']['folder'] + pump_case + '/' +
                  ta_srt_dict['raw_data']['ta_data'][pump_case] % (
                      ta_srt_dict['raw_data']['sample_label'],
                      i,
                  )) as f:
            ta_data_list.append(loadtxt(f))

    ta_data = array(ta_data_list)
    ta_data = ta_data[:, ::-1, :]

    with open(ta_srt_dict['raw_data']['folder'] + pump_case + '/' +
              ta_srt_dict['raw_data']['time_data'][pump_case] %
              (ta_srt_dict['raw_data']['sample_label'], )) as f:
        ta_times = loadtxt(f)[:, 1]

    try:
        os.mkdir(
            '/storage/Reference/Work/University/PhD/TA_Analysis/plots_all_%s_%s'
            % (
                os.path.splitext(os.path.basename(__file__))[0],
                pump_case,
            ))
    except FileExistsError as e:
        pass

    try:
        time_func(
            pool.map,
            functools.partial(
                plot_fit,
                ta_data=ta_data,
                ta_times=ta_times,
                abs_data=abs_data,
                ta_srt_dict=ta_srt_dict,
                pump_case=pump_case,
                colors=colors,
            ),
            range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
                  ta_srt_dict['raw_data']['n_files'][pump_case][0]),
        )
    except FileNotFoundError as e:
        print('Error: %s' % e)
        pool.close()
        exit()

    print('"%s" finished!\n' % pump_case, flush=True)
