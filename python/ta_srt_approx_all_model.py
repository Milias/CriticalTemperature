from common import *


def srt_dist(E, fse, alpha, f_eq, f_zero):
    eq_data = f_eq(E)
    zero_data = f_zero(E)

    eq_data /= trapz(eq_data, E)
    zero_data /= trapz(zero_data, E)

    return eq_data * (1 - alpha) + zero_data * alpha


def f_dist_eq(E, beta, shift):
    return exp(-beta * (E - shift))


def f_dist_zero(E, mu, sigma):
    return stats.norm.pdf(E, loc=mu, scale=sigma)


fit_vars_label = 'fit_vars_model_biexc'
file_version = 'v4'


def TA_model(abs_data, ta_srt_dict, pump_case, steady_data):
    var_list = list(ta_srt_dict[fit_vars_label].keys())

    pump_sigma = ta_srt_dict['raw_data']['pump_sigma'][pump_case]
    pump_mu = ta_srt_dict['raw_data']['pump_mu'][pump_case]

    def model_fun(xdata, *popt, return_dict=False):
        load_popt(popt, globals(), var_list)
        abs_interp = interp1d(
            steady_data[:, 0],
            steady_data[:, -1],
            bounds_error=False,
            fill_value=0.0,
        )(xdata)

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
                mu=pump_mu,
                sigma=pump_sigma,
            ),
        )

        se_hh_data = hh_interp * dist_data
        se_lh_data = lh_interp * dist_data
        se_sum_data = se_hh_data + se_lh_data

        se_sum_data_max = trapz(se_sum_data, xdata)
        se_hh_data /= se_sum_data_max
        se_lh_data /= se_sum_data_max
        se_sum_data /= se_sum_data_max

        se_hh_data *= fse
        se_lh_data *= fse
        se_sum_data *= fse

        depl_data = cont_hh_interp * f_dist_eq(
            xdata,
            sys.d_params.beta,
            shift=xdata[0],
        )
        depl_data /= trapz(depl_data, xdata)
        depl_data *= fdepl

        m_hhX = sys.params.m_hh / (sys.params.m_e + sys.params.m_hh)
        m_eX = sys.params.m_e / (sys.params.m_e + sys.params.m_hh)
        xdata_shift = xdata - abs_shift
        delta_mu = log(sys.params.m_e / sys.params.m_hh) / sys.d_params.beta

        depl2_data = cont_hh_interp * (
            exp(-sys.d_params.beta * m_hhX * xdata_shift) +
            exp(-sys.d_params.beta * (m_eX * xdata_shift - delta_mu)))
        depl2_data /= trapz(depl2_data, xdata)
        depl2_data *= fdepl2

        try:
            hhhh = stats.norm.pdf(
                xdata,
                loc=hhhh_loc,
                scale=hhhh_sig,
            )

            hhhh /= trapz(hhhh, xdata)
            hhhh *= hhhh_mag
        except:
            hhhh = zeros_like(xdata)

        try:
            hhlh = stats.norm.pdf(
                xdata,
                loc=hhlh_loc,
                scale=hhlh_sig,
            )

            hhlh /= trapz(hhlh, xdata)
            hhlh *= hhlh_mag
        except:
            hhlh = zeros_like(xdata)

        result = clip(
            abs_interp - depl_data - depl2_data - se_sum_data + hhhh + hhlh,
            0,
            1,
        )

        if return_dict:
            return {
                'all': result,
                'depl': depl_data,
                'depl2': depl2_data,
                'se_hh': se_hh_data,
                'se_lh': se_lh_data,
                'abs': abs_interp,
                'hhlh': hhlh,
                'hhhh': hhhh,
            }
        else:
            return result

    return model_fun


def plot_fit(
    time_idx,
    loaded_data,
    ta_data,
    abs_data,
    ta_srt_dict,
    pump_case,
    colors,
):
    import matplotlib.pyplot as plt
    matplotlib.use('pdf')

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({
        'font.serif': ['Computer Modern'],
        'text.usetex': False,
    })

    fig_size = (6.8, 5.3)

    n_x, n_y = 1, 1
    fig = plt.figure(figsize=fig_size)
    ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

    eV_min, eV_max = 2.35, 2.65
    E_vec = linspace(eV_min, eV_max, ta_srt_dict['settings']['N_E'])

    ta_times = loaded_data[:, 0]
    popt = loaded_data[time_idx, 1:-1]
    adj_r2 = loaded_data[time_idx, -1]

    time_value = ta_times[time_idx]

    load_popt(popt, globals(), ta_srt_dict[fit_vars_label].keys())

    ta_exp_interp = interp1d(
        ta_data[time_idx, :, 0],
        ta_data[time_idx, :, 1],
    )(E_vec)

    vis_dict = TA_model(
        abs_data,
        ta_srt_dict,
        pump_case,
        ta_data[0, :],
    )(
        E_vec,
        *popt,
        return_dict=True,
    )

    header_list = [
        'hhhh',
        'hhlh',
        'depl',
        'depl2',
        'se_hh',
        'se_lh',
        'abs',
        'all',
    ]
    saved_data = zeros((
        E_vec.size,
        len(header_list) + 2,
    ))

    saved_data[:, 0] = E_vec

    for n_h, key in enumerate(header_list):
        saved_data[:, n_h + 1] = vis_dict[key]

    saved_data[:, -1] = ta_exp_interp

    data_folder = '/storage/Reference/Work/University/PhD/TA_Analysis/data_all_%s_%s_%s/' % (
        os.path.splitext(os.path.basename(__file__))[0],
        pump_case,
        file_version,
    )
    try:
        os.mkdir(data_folder)
    except:
        pass

    savetxt(
        data_folder + '%d.csv' % time_idx,
        saved_data,
        delimiter=',',
        header='E,%s' % ','.join(header_list + ['ta_exp']),
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
        vis_dict['depl'],
        color='m',
        label='N_e N_h',
        linewidth=0.9,
        linestyle='--',
    )

    ax[0].plot(
        E_vec,
        vis_dict['depl2'],
        color='c',
        label='N_e + N_h',
        linewidth=0.9,
        linestyle='--',
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
        E_vec,
        ta_exp_interp,
        color=colors[time_idx],
        linewidth=0.8,
        label='%.2f ps' % time_value,
    )

    text_list = [
        r'fse: %.5f' % fse,
        r'depl: %.5f' % fdepl,
        r'depl2: %.5f' % fdepl2,
    ]

    try:
        text_list.append(r'alpha: %.5f' % alpha)
    except:
        pass

    text_list.extend([
        r'abs_shift: %.2f meV' % (abs_shift * 1e3),
    ])

    try:
        text_list.append(r'hhhh: (%.3f, %.3f eV, %.1f meV)' % (
            hhhh_mag,
            hhhh_loc,
            hhhh_sig * 1e3,
        ))
    except:
        pass

    try:
        text_list.append(r'hhlh: (%.3f, %.3f eV, %.1f meV)' % (
            hhlh_mag,
            hhlh_loc,
            hhlh_sig * 1e3,
        ))
    except:
        pass

    text_list.extend([
        '',
        'Adj R^2: %.5f' % adj_r2,
    ])

    ax[0].text(
        2.3525,
        0.81,
        '\n'.join(text_list),
        fontsize=7,
    )

    ax[0].set_xlim(E_vec[0], E_vec[-1])
    ax[0].set_ylim(0, 1.1)

    ax[0].set_xlabel(r'$E$ (eV)')

    ax[0].legend(
        loc='upper right',
        prop={'size': 8},
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig_folder = '/storage/Reference/Work/University/PhD/TA_Analysis/plots_all_%s_%s_%s/' % (
        os.path.splitext(os.path.basename(__file__))[0],
        pump_case,
        file_version,
    )

    fig_filename = '%03d.png' % time_idx

    try:
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)
    except FileNotFoundError as e:
        pass

    try:
        os.remove(fig_folder + fig_filename)
    except FileNotFoundError as e:
        pass

    fig.savefig(
        fig_folder + fig_filename,
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

    loaded_data = loadtxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/fit_data/popt_%s_%s_%s.csv'
        % ('ta_srt_approx_fits', pump_case, 'v8'),
        delimiter=',',
    )

    try:
        time_func(
            pool.map,
            functools.partial(
                plot_fit,
                loaded_data=loaded_data,
                ta_data=ta_data,
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
