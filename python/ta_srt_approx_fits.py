from common import *
import matplotlib.pyplot as plt
matplotlib.use('pdf')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': ['serif'],
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})


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


file_version = 'v9'

if file_version == 'v2':
    fit_vars_label = 'fit_vars_model_biexc'
    n_x, n_y = 4, 3
elif file_version == 'v3':
    fit_vars_label = 'fit_vars_model'
    n_x, n_y = 3, 2
elif file_version == 'v4':
    fit_vars_label = 'fit_vars_model'
    n_x, n_y = 3, 2
elif file_version == 'v5':
    fit_vars_label = 'fit_vars_model_biexc'
    n_x, n_y = 4, 3
elif file_version == 'v6':
    fit_vars_label = 'fit_vars_model_biexc'
    n_x, n_y = 4, 3
elif file_version == 'v7':
    fit_vars_label = 'fit_vars_model_biexc'
    n_x, n_y = 4, 3
elif file_version == 'v8':
    fit_vars_label = 'fit_vars_model_biexc'
    n_x, n_y = 4, 3
elif file_version == 'v9':
    fit_vars_label = 'fit_vars_model_biexc'
    n_x, n_y = 4, 3


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

        beta = sys.d_params.beta
        dist_data = f_dist_eq(xdata, beta, xdata[0])
        dist_data /= trapz(dist_data, xdata)

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
            beta,
            shift=xdata[0],
        )
        depl_data /= trapz(depl_data, xdata)
        depl_data *= fdepl

        try:
            m_hhX = sys.params.m_hh / (sys.params.m_e + sys.params.m_hh)
            m_eX = sys.params.m_e / (sys.params.m_e + sys.params.m_hh)
            xdata_shift = xdata - abs_shift
            delta_mu = log(
                sys.params.m_e / sys.params.m_hh) / sys.d_params.beta

            depl2_data = cont_hh_interp * (
                exp(-sys.d_params.beta * m_hhX * xdata_shift) +
                exp(-sys.d_params.beta * (m_eX * xdata_shift - delta_mu)))
            depl2_data /= trapz(depl2_data, xdata)
            depl2_data *= fdepl2
        except:
            depl2_data = zeros_like(xdata)

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


def TA_fit(time_idx,
           ta_data,
           abs_data,
           ta_srt_dict,
           pump_case,
           ta_times_zero,
           p0_values=None):
    eV_min, eV_max = 2.35, 2.65
    E_vec = linspace(eV_min, eV_max, ta_srt_dict['settings']['N_E'])

    ta_data_mask = (ta_data[time_idx, :, 0] > eV_min) * (ta_data[time_idx, :,
                                                                 0] < eV_max)

    TA_model_func = TA_model(
        abs_data,
        ta_srt_dict,
        pump_case,
        ta_data[0, :],
    )

    if time_idx > ta_times_zero:
        if p0_values is None:
            p0_values = array([
                ta_srt_dict[fit_vars_label][var]['p0'] if isinstance(
                    ta_srt_dict[fit_vars_label][var]['p0'], float) else
                ta_srt_dict[fit_vars_label][var]['p0'][pump_case]
                for var in ta_srt_dict[fit_vars_label]
            ])
        else:
            n_avg_items = ta_srt_dict['raw_data']['n_avg_items']
            idx_lims = (max(
                time_idx - n_avg_items,
                ta_times_zero,
            ), min(
                time_idx + n_avg_items,
                len(p0_values) - 1,
            ))

            p0_values = sum(
                array(p0_values[idx_lims[0]:idx_lims[1]]),
                axis=0,
            ) / len(p0_values[idx_lims[0]:idx_lims[1]])

        bounds = array([
            tuple(ta_srt_dict[fit_vars_label][var]['bounds']) if isinstance(
                ta_srt_dict[fit_vars_label][var]['bounds'], list) else tuple(
                    ta_srt_dict[fit_vars_label][var]['bounds'][pump_case])
            for var in ta_srt_dict[fit_vars_label]
        ]).T

        try:
            popt, pcov = curve_fit(
                TA_model_func,
                ta_data[time_idx, ta_data_mask, 0],
                ta_data[time_idx, ta_data_mask, 1],
                p0=p0_values,
                bounds=bounds,
                method='trf',
                maxfev=8000,
            )
        except Exception as e:
            print('[Error] %s' % e)
            popt = zeros(len(ta_srt_dict[fit_vars_label].keys()))
            pcov = zeros((popt.size, popt.size))
    else:
        p0_values = array([
            ta_srt_dict[fit_vars_label][var]['p0'] if isinstance(
                ta_srt_dict[fit_vars_label][var]['p0'], float) else
            ta_srt_dict[fit_vars_label][var]['p0'][pump_case]
            for var in ta_srt_dict[fit_vars_label]
        ])

        popt = p0_values
        pcov = zeros((popt.size, popt.size))

    data = {
        'model':
        TA_model_func(ta_data[time_idx, ta_data_mask, 0], *popt),
        'full_model':
        TA_model_func(
            ta_data[time_idx, ta_data_mask, 0],
            *popt,
            return_dict=True,
        ),
        'data':
        ta_data[time_idx, ta_data_mask, 1],
        'n_params':
        len(popt),
    }

    return E_vec, data, popt, pcov


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

fit_results = [[]] * len(ta_srt_dict['settings']['plot_cases'])
popt_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
perr_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
stats_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])

ta_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
ta_times = [[]] * len(ta_srt_dict['settings']['plot_cases'])

ta_diffs = [[]] * len(ta_srt_dict['settings']['plot_cases'])
ta_times_zero = [[]] * len(ta_srt_dict['settings']['plot_cases'])

try:
    os.mkdir('/storage/Reference/Work/University/PhD/TA_Analysis/fit_data')
except:
    pass

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    print(
        '[%d/%d] Processing "%s" ...' % (
            ii + 1,
            len(ta_srt_dict['settings']['plot_cases']),
            pump_case,
        ),
        flush=True,
    )

    ta_data_list = []

    for i in range(*ta_srt_dict['raw_data']['n_files'][pump_case]):
        with open(ta_srt_dict['raw_data']['folder'] + pump_case + '/' +
                  ta_srt_dict['raw_data']['ta_data'][pump_case] % (
                      ta_srt_dict['raw_data']['sample_label'],
                      i,
                  )) as f:
            ta_data_list.append(loadtxt(f))

    ta_data[ii] = array(ta_data_list)
    ta_data[ii] = ta_data[ii][:, ::-1, :]

    with open(ta_srt_dict['raw_data']['folder'] + pump_case + '/' +
              ta_srt_dict['raw_data']['time_data'][pump_case] %
              (ta_srt_dict['raw_data']['sample_label'], )) as f:
        ta_times[ii] = loadtxt(
            f)[ta_srt_dict['raw_data']['n_files'][pump_case][0]:
               ta_srt_dict['raw_data']['n_files'][pump_case][1], 1]

    ta_diffs[ii] = [
        sum((ta_data[ii][0, :, 1] - ta_data[ii][time_idx, :, 1])**2) for
        time_idx in range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
                          ta_srt_dict['raw_data']['n_files'][pump_case][0])
    ]

    q_val = quantile(ta_diffs[ii], 0.05, interpolation='lower')
    ta_times_zero[ii] = flatnonzero(ta_diffs[ii] < q_val)[-1]
    ta_times[ii] -= ta_times[ii][ta_times_zero[ii]]

    fit_results[ii] = time_func(
        pool.map,
        functools.partial(
            TA_fit,
            ta_data=ta_data[ii],
            abs_data=abs_data,
            ta_srt_dict=ta_srt_dict,
            pump_case=pump_case,
            ta_times_zero=ta_times_zero[ii],
        ),
        range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
              ta_srt_dict['raw_data']['n_files'][pump_case][0]),
    )

    popt_arr[ii] = array([r[2] for r in fit_results[ii]])

    n_smooth_passes = ta_srt_dict['raw_data']['n_smooth_passes']
    for jj_smooth in range(n_smooth_passes):
        print(
            '[%d/%d] Smoothing pass' % (
                jj_smooth + 1,
                n_smooth_passes,
            ),
            flush=True,
        )

        fit_results[ii] = time_func(
            pool.map,
            functools.partial(
                TA_fit,
                ta_data=ta_data[ii],
                abs_data=abs_data,
                ta_srt_dict=ta_srt_dict,
                pump_case=pump_case,
                ta_times_zero=ta_times_zero[ii],
                p0_values=popt_arr[ii],
            ),
            range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
                  ta_srt_dict['raw_data']['n_files'][pump_case][0]),
        )

        popt_arr[ii] = array([r[2] for r in fit_results[ii]])

    popt_arr[ii] = array([r[2] for r in fit_results[ii]])
    perr_arr[ii] = array([sqrt(diag(r[3])) for r in fit_results[ii]])
    stats_arr[ii] = array([adj_r_squared(**r[1]) for r in fit_results[ii]])

    saved_data = zeros((
        ta_times[ii].size,
        popt_arr[ii][0, :].size + 2,
    ))

    saved_data[:, 0] = ta_times[ii]
    saved_data[:, 1:-1] = popt_arr[ii]
    saved_data[:, -1] = stats_arr[ii]

    savetxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/fit_data/popt_%s_%s_%s.csv'
        % (
            os.path.splitext(os.path.basename(__file__))[0],
            pump_case,
            file_version,
        ),
        saved_data,
        delimiter=',',
        header='t (ps),%s,adj_r2' %
        ','.join(list(ta_srt_dict[fit_vars_label].keys())),
    )

    saved_data = zeros((
        ta_times[ii].size,
        perr_arr[ii][0, :].size + 1,
    ))

    saved_data[:, 0] = ta_times[ii]
    saved_data[:, 1:] = perr_arr[ii]

    savetxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/fit_data/perr_%s_%s_%s.csv'
        % (
            os.path.splitext(os.path.basename(__file__))[0],
            pump_case,
            file_version,
        ),
        saved_data,
        delimiter=',',
        header='t (ps),%s' %
        ','.join(list(ta_srt_dict[fit_vars_label].keys())),
    )

print(ta_times_zero)

case_colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]

fig_size = (6.8 * 3, 5.3 * 2)
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

fit_vars_list = list(ta_srt_dict[fit_vars_label].keys())
fit_vars_list.append('1 - Adj R$^2$')

for n in range(len(ax)):
    for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
        if n < len(ta_srt_dict[fit_vars_label]):
            ax[n].plot(
                ta_times[ii],
                popt_arr[ii][:, n],
                linewidth=1.6,
                color=case_colors[ii],
                label=r'%s' % pump_case,
            )

            ax[n].fill_between(
                ta_times[ii],
                popt_arr[ii][:, n] + perr_arr[ii][:, n],
                popt_arr[ii][:, n] - perr_arr[ii][:, n],
                color=case_colors[ii],
                alpha=0.2,
            )

            ax[n].set_ylim(
                *ta_srt_dict[fit_vars_label][fit_vars_list[n]]['bounds'])

            ax[n].set_xlim(0, ta_times[ii][-1])
        else:
            ax[n].plot(
                ta_times[ii],
                1 - stats_arr[ii],
                color=case_colors[ii],
                label='%s' % pump_case,
            )

            ax[n].set_yscale('log')
            ax[n].set_xlim(0, ta_times[ii][-1])
            ax[n].set_ylim(2e-5, 5e-3)

        ax[n].axvline(
            x=ta_times[ii][ta_srt_dict['raw_data']['ta_times_zero']
                           [pump_case]],
            color=case_colors[ii],
            linewidth=0.7,
        )

    if n < n_x * (n_y - 1) and fit_vars_list[n] != 'alpha':
        ax[n].xaxis.set_visible(False)
        ax[n].set_xlabel('Time (ps)')

    ax[n].set_xscale('symlog')

    ax[n].axhline(
        y=0,
        color='k',
        linewidth=0.7,
    )

    ax[n].axvline(
        x=0,
        color='k',
        linewidth=0.7,
    )

    ax[n].legend(
        loc='upper right',
        prop={'size': 12},
        title=fit_vars_list[n if n < len(fit_vars_list) else -1].replace(
            '_',
            '\_',
        ),
    )

plt.tight_layout()
fig.subplots_adjust(wspace=0.15, hspace=0.1)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TA_Analysis/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

#plt.show()
