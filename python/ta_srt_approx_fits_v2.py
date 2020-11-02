from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})


def srt_dist(E, fse, alpha, f_eq, f_zero):
    eq_data = f_eq(E)
    zero_data = f_zero(E)

    return eq_data * (1 - alpha) + zero_data * alpha


def f_dist_eq(E, beta, shift):
    return exp(-beta * (E - shift))


def f_dist_zero(E, mu, sigma):
    return stats.norm.pdf(E, loc=mu, scale=sigma)


fit_vars_label = 'fit_vars_model_biexc'


def TA_model(abs_data, ta_srt_dict, pump_case):
    var_list = list(ta_srt_dict[fit_vars_label].keys())

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

        se_sum_data = se_hh_data + se_lh_data

        se_hh_data /= amax(se_sum_data)
        se_lh_data /= amax(se_sum_data)
        se_sum_data /= amax(se_sum_data)

        se_hh_data *= fse
        se_lh_data *= fse
        se_sum_data *= fse

        depl_data = fdepl * (cont_hh_interp + cont_lh_interp)

        hhhh = hhhh_mag * stats.norm.pdf(
            xdata - abs_shift,
            loc=hhhh_loc - abs_shift,
            scale=hhhh_sig,
        )

        hhlh = hhlh_mag * stats.norm.pdf(
            xdata - abs_shift,
            loc=hhlh_loc - abs_shift,
            scale=hhlh_sig,
        )

        result = abs_interp - depl_data - se_sum_data + hhhh + hhlh
        clip(result, 0, 1)

        if return_dict:
            return {
                'all': result,
                'depl': depl_data,
                'se_hh': se_hh_data,
                'se_lh': se_lh_data,
                'abs': abs_interp,
                'hhlh': hhlh,
                'hhhh': hhhh,
            }
        else:
            return result

    return model_fun


def TA_fit(time_idx, ta_data, abs_data, ta_srt_dict, pump_case):
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
        ta_srt_dict[fit_vars_label][var]['p0'] if isinstance(
            ta_srt_dict[fit_vars_label][var]['p0'], float) else
        ta_srt_dict[fit_vars_label][var]['p0'][pump_case]
        for var in ta_srt_dict[fit_vars_label]
    ])

    bounds = array([
        tuple(ta_srt_dict[fit_vars_label][var]['bounds']) if isinstance(
            ta_srt_dict[fit_vars_label][var]['bounds'], list) else tuple(
                ta_srt_dict[fit_vars_label][var]['bounds'][pump_case])
        for var in ta_srt_dict[fit_vars_label]
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

fit_results = [[] * len(ta_srt_dict['settings']['plot_cases'])]

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    print(
        '[%d/%d] Processing "%s" ...' % (
            ii + 1,
            len(ta_srt_dict['settings']['plot_cases']),
            pump_case,
        ),
        flush=True,
    )
    fit_results[ii] = time_func(
        pool.map,
        functools.partial(
            TA_fit,
            ta_data=ta_data,
            abs_data=abs_data,
            ta_srt_dict=ta_srt_dict,
            pump_case=pump_case,
        ),
        range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
              ta_srt_dict['raw_data']['n_files'][pump_case][0]),
    )

popt_arr = array([r[2] for r in fit_results])
perr_arr = array([sqrt(diag(r[3])) for r in fit_results])
stats_arr = array([adj_r_squared(**r[1]) for r in fit_results])
"""
ta_times_zero = 42
ta_times_last = ta_times.size
time_slice = ta_times[ta_times_zero:ta_times_last]
alpha_slice = popt_arr[ta_times_zero:ta_times_last, 2]

tau_p0_values = tuple([
    ta_srt_dict['alpha_fit_v2'][var]['p0']
    for var in ta_srt_dict['alpha_fit_v2']
])

tau_bounds = array([
    tuple(ta_srt_dict['alpha_fit_v2'][var]['bounds'])
    for var in ta_srt_dict['alpha_fit_v2']
]).T


def tau_model_func(t, tau_vec, v_vec):
    return [exp(-(t + v) / tau) for tau, v in zip(tau_vec, v_vec)]


def tau_model(t, tau0, tau1, tau2, v0, v1, v2, split=False):
    local_dict = locals()
    n_vars = len(tuple(filter(lambda x: x[:3] == 'tau', local_dict.keys())))
    data = tau_model_func(
        t - t[0],
        [local_dict['tau%d' % ii] for ii in range(n_vars)],
        [local_dict['v%d' % ii] for ii in range(n_vars)],
    )

    if split:
        return data
    else:
        return sum(data, axis=0)


try:
    tau_popt, tau_pcov = curve_fit(
        tau_model,
        time_slice,
        alpha_slice,
        p0=tau_p0_values,
        bounds=tau_bounds,
        method='trf',
    )
except:
    tau_popt = tau_p0_values
    tau_pcov = zeros(2 * [len(tau_p0_values)])

sorted_idx_list = sorted(range(3), key=lambda ii: tau_popt[ii])
"""

case_colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]

fig_size = (6.8 * 3, 5.3 * 2)

n_x, n_y = 4, 3
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

for n in range(len(ax)):
    if n < len(fit_vars_list):
        ax[n].plot(
            ta_times,
            popt_arr[:, n],
            linewidth=1.6,
            color='m',
            label=(r'$%s$ (%s)' if fit_var_units[n] != '' else r'$%s$%s') %
            (fit_vars_list[n].replace('_', '\_'), fit_var_units[n]),
        )

        ax[n].xaxis.set_visible(False)

        if amax(popt_arr[ta_times_zero:, n]) < 1:
            ax[n].axhline(
                y=0,
                color='k',
                linewidth=0.7,
                linestyle='--',
            )
            ax[n].set_ylim(
                None,
                1.1 * amax(popt_arr[ta_times_zero:, n]),
            )

        if amax(popt_arr[ta_times_zero:, n]) > 1e2:
            ax[n].set_yscale('log')
        """
        if n == 2:
            data = tau_model(
                time_slice,
                *tau_popt,
                split=True,
            )
            for ii in range(3):
                ax[n].plot(
                    time_slice,
                    data[sorted_idx_list[ii]],
                    color=fit_colors[sorted_idx_list[ii]],
                    linestyle='--',
                    linewidth=1.4,
                    label=(r'$\tau_{%%d}: %%.%df\pm%%.%df$ ps' %
                           tuple(2 * [(2, 0, 0)[ii]])) % (
                               ii,
                               tau_popt[sorted_idx_list[ii]],
                               sqrt(diag(tau_pcov)[sorted_idx_list[ii]]),
                           ),
                )

            ax[n].plot(
                time_slice,
                sum(data, axis=0),
                color='k',
                linestyle='-',
                linewidth=2.0,
            )

            print(tau_popt)
        """
    else:
        ax[n].plot(
            ta_times,
            1 - stats_arr[:, 0],
            color='r',
            label='$1$ $-$ Adj R$^2$',
        )

        ax[n].set_yscale('log')

    ax[n].set_xscale('symlog')

    ax[n].axvline(
        x=0,
        color='k',
        linewidth=0.7,
    )

    ax[n].set_xlim(ta_times[0], ta_times[-1])

    ax[n].legend(
        loc='upper left',
        prop={'size': 12},
    )

ax[-1].set_xlabel('Time (ps)')

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TA_Analysis/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v2'),
    transparent=True,
)

plt.show()
