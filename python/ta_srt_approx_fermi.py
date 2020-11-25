from common import *
import matplotlib.pyplot as plt

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


def depl_frac(E_F, sys):
    return exp(-sys.d_params.beta * E_F)


file_version = 'v6'
fit_vars_label = 'fit_vars_model_biexc'
n_x, n_y = 1, 3


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
        )(xdata)

        cont_hh_interp /= amax(cont_hh_interp)

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

        se_sum_data_max = amax(se_sum_data)
        se_hh_data /= se_sum_data_max
        se_lh_data /= se_sum_data_max
        se_sum_data /= se_sum_data_max

        se_hh_data *= fse
        se_lh_data *= fse
        se_sum_data *= fse

        depl_data = fdepl * cont_hh_interp

        try:
            hhhh = stats.norm.pdf(
                xdata,
                loc=hhhh_loc,
                scale=hhhh_sig,
            )

            hhhh /= amax(hhhh)
            hhhh *= hhhh_mag
        except:
            hhhh = zeros_like(xdata)

        try:
            hhlh = stats.norm.pdf(
                xdata,
                loc=hhlh_loc,
                scale=hhlh_sig,
            )

            hhlh /= amax(hhlh)
            hhlh *= hhlh_mag
        except:
            hhlh = zeros_like(xdata)

        try:
            cont_abs = interp1d(
                abs_data[:, 0],
                abs_data[:, 3],
                bounds_error=False,
                fill_value=0.0,
            )(xdata - cont_loc)

            cont_abs /= amax(cont_abs)
            cont_abs *= fdepl
        except:
            cont_abs = zeros_like(xdata)

        result = clip(
            abs_interp - depl_data - se_sum_data + hhhh + hhlh + cont_abs, 0,
            1)

        if return_dict:
            return {
                'all': result,
                'depl': depl_data,
                'se_hh': se_hh_data,
                'se_lh': se_lh_data,
                'abs': abs_interp,
                'hhlh': hhlh,
                'hhhh': hhhh,
                'cont_abs': cont_abs,
            }
        else:
            return result

    return model_fun


def numerical_fraction(
    time_idx,
    popt,
    abs_data,
    ta_data,
    ta_srt_dict,
    pump_case,
):
    eV_min, eV_max = 2.35, 2.65
    E_vec = linspace(eV_min, eV_max, ta_srt_dict['settings']['N_E'])

    vis_dict = TA_model(
        abs_data,
        ta_srt_dict,
        pump_case,
        ta_data[0, :],
    )(
        E_vec,
        *popt[time_idx],
        return_dict=True,
    )

    load_popt(popt[time_idx], globals(), ta_srt_dict[fit_vars_label].keys())

    if fdepl > 0:
        return trapz(
            vis_dict['depl'] / fdepl - vis_dict['cont_abs'] / fdepl,
            E_vec,
        )

    return 0


def numerical_fraction_exc(
    time_idx,
    popt,
    abs_data,
    ta_data,
    ta_srt_dict,
    pump_case,
):
    eV_min, eV_max = 2.35, 2.65
    E_vec = linspace(eV_min, eV_max, ta_srt_dict['settings']['N_E'])

    vis_dict = TA_model(
        abs_data,
        ta_srt_dict,
        pump_case,
        ta_data[0, :],
    )(
        E_vec,
        *popt[time_idx],
        return_dict=True,
    )

    load_popt(popt[time_idx], globals(), ta_srt_dict[fit_vars_label].keys())

    if fse > 0:
        return trapz(
            vis_dict['se_hh'] / fse + vis_dict['se_lh'] / fse,
            E_vec,
        )

    return 0


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

popt_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
ta_times = [[]] * len(ta_srt_dict['settings']['plot_cases'])
frac_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
num_frac_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
num_frac_exc_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])

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

    saved_data = loadtxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/fit_data/popt_%s_%s_%s.csv'
        % (
            'ta_srt_approx_fits',
            pump_case,
            file_version,
        ),
        delimiter=',',
    )

    ta_times[ii] = saved_data[:, 0]
    popt_arr[ii] = saved_data[:, 1:-1]

    frac_data[ii] = depl_frac(popt_arr[ii][:, 7], sys)

    num_frac_exc_data[ii] = time_func(
        pool.map,
        functools.partial(
            numerical_fraction_exc,
            popt=popt_arr[ii],
            abs_data=abs_data,
            ta_data=ta_data,
            ta_srt_dict=ta_srt_dict,
            pump_case=pump_case,
        ),
        range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
              ta_srt_dict['raw_data']['n_files'][pump_case][0]),
    )

    num_frac_exc_data[ii] /= num_frac_exc_data[ii][ta_srt_dict['raw_data']['ta_times_zero'][pump_case]]

    num_frac_data[ii] = time_func(
        pool.map,
        functools.partial(
            numerical_fraction,
            popt=popt_arr[ii],
            abs_data=abs_data,
            ta_data=ta_data,
            ta_srt_dict=ta_srt_dict,
            pump_case=pump_case,
        ),
        range(ta_srt_dict['raw_data']['n_files'][pump_case][1] -
              ta_srt_dict['raw_data']['n_files'][pump_case][0]),
    )

    num_frac_data[ii] /= amax(num_frac_data[ii][183])

try:
    os.mkdir('/storage/Reference/Work/University/PhD/TA_Analysis/fit_data')
except:
    pass

case_colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]

fig_size = (6.8, 5.3 * 2)
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

fit_vars_list = ['fse', 'cont_loc', 'n_q']

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    ax[0].plot(
        ta_times[ii],
        popt_arr[ii][:, 0],
        linewidth=1.6,
        color=case_colors[ii],
        label=r'%s' % pump_case,
    )

    ax[1].plot(
        ta_times[ii],
        popt_arr[ii][:, 7],
        linewidth=1.6,
        color=case_colors[ii],
        label=r'%s' % pump_case,
    )

    ax[2].plot(
        ta_times[ii],
        frac_data[ii],
        linewidth=1.6,
        color=case_colors[ii],
        label=r'%s' % pump_case,
    )
    """
    ax[2].plot(
        ta_times[ii],
        num_frac_data[ii],
        linewidth=1.0,
        linestyle='-',
        color=case_colors[ii],
    )
    """

    for n in range(len(ax)):
        ax[n].set_xscale('symlog')
        ax[n].set_xlim(0, ta_times[ii][-1])

        ax[n].axvline(
            x=2.0,
            color='g',
            linewidth=0.6,
        )

        ax[n].axvline(
            x=ta_times[ii][ta_srt_dict['raw_data']['ta_times_zero']
                           [pump_case]],
            color=case_colors[ii],
            linewidth=0.7,
        )

        if n < len(fit_vars_list) - 1:
            ax[n].set_ylim(
                *ta_srt_dict[fit_vars_label][fit_vars_list[n]]['bounds'])

        ax[2].set_ylim(0.0, 0.4)
        ax[2].axhline(
            y=0.0,
            color='k',
            linewidth=0.7,
        )

        if n < n_x * (n_y - 1):
            ax[n].xaxis.set_visible(False)
        else:
            ax[n].set_xlabel('Time (ps)')

for n in range(len(ax)):
    ax[n].legend(
        loc='upper right',
        prop={'size': 12},
        title=fit_vars_list[n if n < len(fit_vars_list) else -1].replace(
            '_',
            '\_',
        ),
    )

plt.tight_layout()
fig.subplots_adjust(wspace=0.15, hspace=0.05)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TA_Analysis/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()
