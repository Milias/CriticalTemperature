from common import *
import matplotlib.pyplot as plt
matplotlib.use('pdf')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': ['serif'],
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})


def dmu_X(n_X, sys):
    p0 = pi * sys.c_hbarc**2 * 0.5 / (sys.params.m_e +
                                      sys.params.m_hh) / sys.c_m_e
    exponent = p0 * n_X

    if exponent < 700:
        return p0 * exp(exponent) / (exp(exponent) - 1)
    else:
        return p0


def dmu_q(n_q, sys):
    p_e = pi * sys.c_hbarc**2 * sys.d_params.beta / sys.params.m_e / sys.c_m_e * n_q
    p_hh = pi * sys.c_hbarc**2 * sys.d_params.beta / sys.params.m_hh / sys.c_m_e * n_q

    r = 0

    if p_e < 700:
        exp_e = exp(p_e)
        r += exp_e / sys.params.m_e / sys.c_m_e / (exp_e - 1)
    else:
        r += 1 / sys.params.m_e / sys.c_m_e

    if p_hh < 700:
        exp_hh = exp(p_hh)
        r += exp_hh / sys.params.m_hh / sys.c_m_e / (exp_hh - 1)
    else:
        r += 1 / sys.params.m_hh / sys.c_m_e

    return pi * sys.c_hbarc**2 * r


def dn_dt(
    t,
    y_vec,
    sys,
    E_X,
    r_q,
    tau_mu,
    tau_qt,
    tau_Xt,
    tau_decay,
    chi_t_total,
    tau_pump,
    pump_shift,
    n_gamma_0,
):
    n_q, n_X, n_t = y_vec

    mu_e = sys.mu_ideal(n_q)
    mu_h = sys.mu_h_ideal(n_q)
    mu_q = mu_e + mu_h

    mu_X = sys.mu_exc(n_X, E_X)

    dmu = 0
    if n_q > 1e-10:
        dmu += 0.5 / dmu_q(n_q, sys)
    if n_X > 1e-10:
        dmu += 0.5 / dmu_X(n_X, sys)

    dn_pump = n_gamma_0 * stats.norm.pdf(
        t,
        loc=pump_shift,
        scale=tau_pump,
    )

    return [
        (-mu_q / tau_mu + mu_X / tau_mu) * dmu + r_q * dn_pump -
        (chi_t_total - n_t / n_gamma_0) * n_q / tau_qt - n_q / tau_decay,
        (-mu_X / tau_mu + mu_q / tau_mu) * dmu + (1.0 - r_q) * dn_pump -
        (chi_t_total - n_t / n_gamma_0) * n_X / tau_Xt - n_X / tau_decay,
        (n_q / tau_qt + n_X / tau_Xt) * (chi_t_total - n_t / n_gamma_0),
    ]


def solve_chem_eq(t_span, y0_vec, args_vec):
    return solve_ivp(
        dn_dt,
        t_span,
        y0_vec,
        args=args_vec,
        dense_output=True,
    )


file_version = 'v10'
fit_vars_label = 'fit_vars_model_biexc'
c_fit_label = 'fit_chem_eq_pump'
n_x, n_y = 2, 2


def fit_ivp_model(t_span, ravel=True, n_q_0=1e-10, n_X_0=1e-10):
    var_list = list(ta_srt_dict[c_fit_label].keys())

    #tau_pump = ta_srt_dict['raw_data']['tau_pump']
    #pump_shift = ta_srt_dict['raw_data']['pump_shift']

    def fit_ivp_func(xdata, *popt):
        load_popt(popt, globals(), var_list)

        #print(popt)

        args_vec = [
            sys,
            -135e-3,
            r_q,
            tau_mu,
            tau_qt,
            tau_Xt,
            tau_decay,
            chi_t_total,
            tau_pump,
            pump_shift,
            n_gamma_0,
        ]
        result = solve_chem_eq(
            t_span,
            [n_q_0, n_X_0, 0.0],
            args_vec,
        )

        if ravel:
            n_x = xdata.size // 2
        else:
            n_x = xdata.size

        n_arr = result.sol(xdata[:n_x])

        n_arr[0] *= sigma_q

        if ravel:
            return n_arr[:2].ravel()
        else:
            return n_arr

    return fit_ivp_func


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

popt_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
ta_times = [[]] * len(ta_srt_dict['settings']['plot_cases'])

n_result_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
abs_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
t_vec = [[]] * len(ta_srt_dict['settings']['plot_cases'])

popt_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
perr_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])

fit_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
fit_result_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])

c_fit_popt_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
c_fit_pcov_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
c_fit_perr_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])

plot_t_vec = [[]] * len(ta_srt_dict['settings']['plot_cases'])

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

    t0_idx = ta_srt_dict['raw_data']['ta_times_start'][pump_case]
    end_n_t = 163

    ta_times[ii] = saved_data[:, 0]
    ta_times[ii] -= ta_times[ii][t0_idx]
    popt_arr[ii] = saved_data[:, 1:]

    saved_data = loadtxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/fit_data/perr_%s_%s_%s.csv'
        % (
            'ta_srt_approx_fits',
            pump_case,
            file_version,
        ),
        delimiter=',',
    )

    perr_arr[ii] = saved_data[:, 1:]

    t_vec[ii] = ta_times[ii][t0_idx:end_n_t]

    t_span = [t_vec[ii][0], t_vec[ii][-1]]

    popt_arr[ii][:, 1] += popt_arr[ii][:, 2]

    fit_data[ii] = array([
        popt_arr[ii][t0_idx:end_n_t, 1],
        popt_arr[ii][t0_idx:end_n_t, 0],
    ]).ravel()

    sigma_data = array([
        perr_arr[ii][t0_idx:end_n_t, 1],
        perr_arr[ii][t0_idx:end_n_t, 0],
    ]).ravel()

    p0_values = array([
        ta_srt_dict[c_fit_label][var]['p0'] if isinstance(
            ta_srt_dict[c_fit_label][var]['p0'], float) else
        ta_srt_dict[c_fit_label][var]['p0'][pump_case]
        for var in ta_srt_dict[c_fit_label]
    ])

    bounds = array([
        tuple(ta_srt_dict[c_fit_label][var]['bounds']) if isinstance(
            ta_srt_dict[c_fit_label][var]['bounds'], list) else tuple(
                ta_srt_dict[c_fit_label][var]['bounds'][pump_case])
        for var in ta_srt_dict[c_fit_label]
    ]).T

    fit_func = fit_ivp_model(
        t_span,
        #n_q_0=popt_arr[ii][t0_idx, 1],
        #n_X_0=popt_arr[ii][t0_idx, 0],
    )

    c_fit_popt_arr[ii], c_fit_pcov_arr[ii] = time_func(
        curve_fit,
        fit_func,
        array([t_vec[ii], t_vec[ii]]).ravel(),
        fit_data[ii],
        sigma=sigma_data + 1e-5,
        p0=p0_values,
        bounds=bounds,
        method='trf',
        maxfev=10000,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
    )

    print(c_fit_popt_arr[ii])

    fit_func = fit_ivp_model(
        t_span,
        ravel=False,
        #n_q_0=popt_arr[ii][t0_idx, 1],
        #n_X_0=popt_arr[ii][t0_idx, 0],
    )
    fit_result_data[ii] = fit_func(t_vec[ii], *c_fit_popt_arr[ii])

    c_fit_perr_arr[ii] = sqrt(diag(c_fit_pcov_arr[ii]))

    plot_t_vec[ii] = linspace(0.0, ta_times[ii][end_n_t], 1 << 10)
    plot_func = fit_ivp_model(
        [plot_t_vec[ii][0], plot_t_vec[ii][-1]],
        ravel=False,
        #n_q_0=popt_arr[ii][t0_idx, 1],
        #n_X_0=popt_arr[ii][t0_idx, 0],
    )

    n_result_data[ii] = plot_func(plot_t_vec[ii], *c_fit_popt_arr[ii])

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    saved_data = zeros((
        t_vec[ii].size,
        4,
    ))

    saved_data[:, 0] = t_vec[ii]
    saved_data[:, 1:] = fit_result_data[ii].T

    savetxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/c_fit_data/fit_%s_%s_%s.csv'
        % (
            os.path.splitext(os.path.basename(__file__))[0],
            pump_case,
            file_version,
        ),
        saved_data,
        delimiter=',',
        header='t (ps),n_q,n_X,n_t',
    )

    saved_data = zeros((
        plot_t_vec[ii].size,
        4,
    ))

    saved_data[:, 0] = plot_t_vec[ii]
    saved_data[:, 1:] = n_result_data[ii].T

    savetxt(
        '/storage/Reference/Work/University/PhD/TA_Analysis/c_fit_data/plot_%s_%s_%s.csv'
        % (
            os.path.splitext(os.path.basename(__file__))[0],
            pump_case,
            file_version,
        ),
        saved_data,
        delimiter=',',
        header='t (ps),n_q,n_X,n_t',
    )

saved_data = zeros((
    len(ta_srt_dict['settings']['plot_cases']),
    2 * len(ta_srt_dict[c_fit_label].keys()),
))

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    saved_data[ii, :saved_data.shape[1] // 2] = c_fit_popt_arr[ii]
    saved_data[ii, saved_data.shape[1] // 2:] = c_fit_perr_arr[ii]

savetxt(
    '/storage/Reference/Work/University/PhD/TA_Analysis/c_fit_data/popt_%s_%s.csv'
    % (
        os.path.splitext(os.path.basename(__file__))[0],
        file_version,
    ),
    saved_data,
    delimiter=',',
    header='%s,%s' % (
        ','.join(list(ta_srt_dict[c_fit_label].keys())),
        ','.join(
            ['err_%s' % var for var in list(ta_srt_dict[c_fit_label].keys())]),
    ),
)

try:
    os.mkdir('/storage/Reference/Work/University/PhD/TA_Analysis/fit_data')
except:
    pass

case_colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]


def gen_legend(ii, text, index_list, digit_list):
    var_list = list(ta_srt_dict[c_fit_label].keys())
    pump_case = ta_srt_dict['settings']['plot_cases'][ii]

    return ('%s: %s\n%s' % (
        pump_case,
        text,
        '\n'.join([('%%s: %%.%df$\pm$%%.%df%s' % (
            digit_list[jj],
            digit_list[jj],
            (' %s' % ta_srt_dict[c_fit_label][var_list[idx]]['unit'])
            if ta_srt_dict[c_fit_label][var_list[idx]]['unit'] != '' else '',
        )) % (
            var_list[idx],
            c_fit_popt_arr[ii][idx],
            c_fit_perr_arr[ii][idx],
        ) for jj, idx in enumerate(index_list)]),
    )).replace('_', '\_')


fig_size = (6.8 * 3, 5.3 * 2)
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

fit_vars_list = ['$n_X$', '$n_q$', '$n_t$', '$n_\gamma$']

for ii, pump_case in enumerate(ta_srt_dict['settings']['plot_cases']):
    ax[0].plot(
        ta_times[ii],
        popt_arr[ii][:, 0],
        color=case_colors[ii],
        linestyle='--',
        linewidth=0.9,
        label='%s: TA fit' % pump_case,
    )

    ax[1].plot(
        ta_times[ii],
        popt_arr[ii][:, 1],
        color=case_colors[ii],
        linestyle='--',
        linewidth=0.9,
        label='%s: TA fit' % pump_case,
    )

    ax[0].fill_between(
        ta_times[ii],
        popt_arr[ii][:, 0] + perr_arr[ii][:, 0],
        popt_arr[ii][:, 0] - perr_arr[ii][:, 0],
        color=case_colors[ii],
        alpha=0.2,
    )

    ax[1].fill_between(
        ta_times[ii],
        popt_arr[ii][:, 1] + perr_arr[ii][:, 1],
        popt_arr[ii][:, 1] - perr_arr[ii][:, 1],
        color=case_colors[ii],
        alpha=0.2,
    )

    ax[0].plot(
        plot_t_vec[ii],
        n_result_data[ii][1],
        color=case_colors[ii],
        linestyle='-',
        linewidth=1.6,
        label=gen_legend(
            ii,
            'CE fit',
            (1, 2, 5),
            (4, 2, 4),
        ),
    )

    ax[1].plot(
        plot_t_vec[ii],
        n_result_data[ii][0],
        color=case_colors[ii],
        linestyle='-',
        linewidth=1.6,
        label=gen_legend(
            ii,
            'CE fit',
            (3, 4, 9),
            (2, 4, 1),
        ),
    )

    ax[2].plot(
        plot_t_vec[ii],
        n_result_data[ii][2],
        color=case_colors[ii],
        linestyle='-',
        linewidth=1.5,
        label=gen_legend(
            ii,
            'CE fit',
            (8, ),
            (5, ),
        ),
    )

    ax[3].plot(
        plot_t_vec[ii],
        c_fit_popt_arr[ii][0] * stats.norm.cdf(
            plot_t_vec[ii],
            loc=c_fit_popt_arr[ii][7],
            scale=c_fit_popt_arr[ii][6],
        ),
        color=case_colors[ii],
        linestyle='-',
        linewidth=1.5,
        label=gen_legend(
            ii,
            'CE fit',
            (0, 6, 7),
            (4, 4, 4),
        ),
    )

    for n in range(len(ax)):
        ax[n].set_xlim(0.0, t_vec[ii][-1])
        ax[n].set_xscale('symlog')

        ax[n].axvline(
            x=c_fit_popt_arr[ii][7],
            color=case_colors[ii],
            linewidth=0.6,
        )

        ax[n].axvline(
            x=2.0,
            color='g',
            linewidth=0.6,
        )

        if n < n_x * (n_y - 1):
            ax[n].xaxis.set_visible(False)
        else:
            ax[n].set_xlabel('Time (ps)')

for n in range(len(ax)):
    ax[n].legend(
        loc='upper left',
        prop={'size': 12},
        title=fit_vars_list[n],
    )

plt.tight_layout()
fig.subplots_adjust(wspace=0.1, hspace=0.0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TA_Analysis/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

#plt.show()
