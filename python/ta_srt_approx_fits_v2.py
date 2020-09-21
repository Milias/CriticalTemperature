from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = (6.8 * 3, 5.3 * 2)

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

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]


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

p0_values = tuple(
    [ta_srt_dict['fit_vars'][var]['p0'] for var in ta_srt_dict['fit_vars']])

fit_var_units = tuple(
    [ta_srt_dict['fit_vars'][var]['unit'] for var in ta_srt_dict['fit_vars']])

bounds = array([
    tuple(ta_srt_dict['fit_vars'][var]['bounds'])
    for var in ta_srt_dict['fit_vars']
]).T


def model(ta_fit_abs_data, E_vec, srt_time):
    E_max = ta_fit_abs_data[ta_fit_abs_data[:, -1].argmax(), 0]

    ta_fit_abs_interp = interp1d(
        ta_fit_abs_data[:, 0],
        ta_fit_abs_data[:, -1],
    )
    abs_data = ta_fit_abs_interp(E_vec)

    def model_fun(xdata, *popt):
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

        hhhh = hhhh_mag * stats.norm.pdf(
            E_vec,
            loc=hhhh_loc,
            scale=hhhh_sig,
        )

        hhlh = hhlh_mag * stats.norm.pdf(
            E_vec,
            loc=hhlh_loc,
            scale=hhlh_sig,
        )

        return abs_data * (1 - fdepl - fse * dist_data) + hhhh + hhlh

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

    E_vec = linspace(eV_min, eV_max, 1 << 8)

    ta_data_mask = (ta_data[time_idx, :, 0] > eV_min) * (ta_data[time_idx, :,
                                                                 0] < eV_max)

    model_func = model(
        loaded_abs_data[2],
        ta_data[time_idx, ta_data_mask, 0],
        time_value,
    )

    try:
        popt, pcov = curve_fit(
            model_func,
            ta_data[time_idx, ta_data_mask, 0],
            ta_data[time_idx, ta_data_mask, 1],
            p0=p0_values,
            bounds=bounds,
            method='trf',
        )
    except:
        popt = p0_values
        pcov = zeros(2 * [len(p0_values)])

    model_data = model_func(ta_data[time_idx, ta_data_mask, 0], *popt)

    data_stats = (
        adj_r_squared(
            ta_data[time_idx, ta_data_mask, 1],
            model_data,
        ),
        aic_criterion(
            ta_data[time_idx, ta_data_mask, 1],
            model_data,
        ),
    )

    return popt, pcov, data_stats


args_list = map(lambda i: (
    i,
    ta_data,
    loaded_abs_data[2],
), range(ta_times.size))

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
fit_results = time_func(pool.starmap, full_fit_at_time_idx, args_list)

popt_arr = array([r[0] for r in fit_results])
perr_arr = array([sqrt(diag(r[1])) for r in fit_results])
stats_arr = array([r[2] for r in fit_results])

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

n_x, n_y = 4, 3
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

fit_colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]

sorted_idx_list = sorted(range(3), key=lambda ii: tau_popt[ii])

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
        """
        ax[n].fill_between(
            ta_times,
            popt_arr[:, n] - perr_arr[:, n],
            popt_arr[:, n] + perr_arr[:, n],
            color='r',
            alpha=0.3,
        )
        """

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
        """
        else:
            ax[n].set_ylim(
                0.9 * amin(popt_arr[ta_times_zero:, n]),
                1.1 * amax(popt_arr[ta_times_zero:, n]),
            )
        """

        if amax(popt_arr[ta_times_zero:, n]) > 1e2:
            ax[n].set_yscale('log')

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
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()
