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


def dn_dt(t, y_vec, sys, E_X, chi_t_total, tau_mu, tau_qt, tau_Xt):
    n_q, n_X, n_t = y_vec

    mu_e = sys.mu_ideal(n_q)
    mu_h = sys.mu_h_ideal(n_q)
    mu_q = mu_e + mu_h

    mu_X = sys.mu_exc(n_X, E_X)

    dmu = 0.5 * (1 / dmu_q(n_q, sys) + 1 / dmu_X(n_X, sys))

    n_total = n_q + n_X + n_t

    return [
        (-mu_q / tau_mu + mu_X / tau_mu) * dmu - n_q *
        (chi_t_total - n_t / n_total) / tau_qt,
        (-mu_X / tau_mu + mu_q / tau_mu) * dmu - n_X *
        (chi_t_total - n_t / n_total) / tau_Xt,
        (chi_t_total - n_t / n_total) * (n_q / tau_qt + n_X / tau_Xt),
    ]


def solve_chem_eq(t_span, y0_vec, args_vec):
    return solve_ivp(
        dn_dt,
        t_span,
        y0_vec,
        args=args_vec,
        dense_output=True,
    )


file_version = 'v7'
fit_vars_label = 'fit_vars_model_biexc'
c_fit_label = 'fit_chem_eq'
n_x, n_y = 2, 2

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

#pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

popt_arr = [[]] * len(ta_srt_dict['settings']['plot_cases'])
ta_times = [[]] * len(ta_srt_dict['settings']['plot_cases'])
n_result_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
abs_data = [[]] * len(ta_srt_dict['settings']['plot_cases'])
t_vec = [[]] * len(ta_srt_dict['settings']['plot_cases'])

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
    ta_times[ii] -= ta_times[ii][ta_srt_dict['raw_data']['ta_times_zero']
                                 [pump_case]]
    popt_arr[ii] = saved_data[:, 1:-1]

    end_n_t = 143

    for jj in [0, 3]:
        popt_arr[ii][:, jj] /= popt_arr[ii][end_n_t, jj]

    t_span = [
        ta_times[0][ta_srt_dict['raw_data']['ta_times_zero']['HH']],
        ta_times[0][end_n_t],
    ]
    y0_vec = [
        ta_srt_dict[c_fit_label][n_label]['p0'][pump_case]
        for n_label in ['n_q_0', 'n_X_0']
    ]

    y0_vec.append(0)

    args_vec = [
        sys,
        -135e-3,
        ta_srt_dict[c_fit_label]['chi_t_total']['p0'],
        ta_srt_dict[c_fit_label]['tau_mu']['p0'],
        ta_srt_dict[c_fit_label]['tau_qt']['p0'],
        ta_srt_dict[c_fit_label]['tau_Xt']['p0'],
    ]

    t_vec[ii] = linspace(t_span[0], t_span[1], 1 << 8)
    n_result_data[ii] = solve_chem_eq(t_span, y0_vec, args_vec).sol(t_vec[ii])

    abs_data[ii] = array([
        n_result_data[ii][0],
        n_result_data[ii][1],
    ])

    """
    abs_data[ii] /= repeat(
        abs_data[ii][:, -1].reshape(-1, 1),
        t_vec[ii].size,
        axis=1,
    )

    n_result_data[ii] /= repeat(
        n_result_data[ii][:, -1].reshape(-1, 1),
        t_vec[ii].size,
        axis=1,
    )
    """

try:
    os.mkdir('/storage/Reference/Work/University/PhD/TA_Analysis/fit_data')
except:
    pass

n_label_vec = ['$n_q$', '$n_X$', '$n_t$']

case_colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 3)
]

fig_size = (6.8 * 2, 5.3 * 2)
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

fit_vars_list = ['fse', 'fdepl']

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
        popt_arr[ii][:, 3],
        linewidth=1.6,
        color=case_colors[ii],
        label=r'%s' % pump_case,
    )

    ax[2].plot(
        t_vec[ii],
        n_result_data[ii][2],
        color=case_colors[ii],
        linestyle='--',
        linewidth=1.6,
    )

    ax[2].plot(
        t_vec[ii],
        n_result_data[ii][1],
        color=case_colors[ii],
        label=r'%s' % pump_case,
        linestyle='-',
        linewidth=1.6,
    )

    ax[3].plot(
        t_vec[ii],
        n_result_data[ii][0],
        color=case_colors[ii],
        label=r'%s' % pump_case,
        linewidth=1.6,
    )

    for n in range(len(ax)):
        ax[n].set_xlim(t_vec[0][0], t_vec[0][-1])
        if n > 2:
            ax[n].set_yscale('log')

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

        if n < n_x * (n_y - 1):
            ax[n].xaxis.set_visible(False)
        else:
            ax[n].set_xlabel('Time (ps)')

for n in range(len(ax)):
    if n < len(fit_vars_list):
        ax[n].legend(
            loc='upper right',
            prop={'size': 12},
            title=fit_vars_list[n],
        )
    else:
        ax[n].legend(
            loc='upper right',
            prop={'size': 12},
            title=n_label_vec[-n + 3],
        )

plt.tight_layout()
fig.subplots_adjust(wspace=0.15, hspace=0.0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TA_Analysis/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()
