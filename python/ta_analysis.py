from common import *


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


def model(ta_fit_abs_data, E_vec):
    ta_fit_abs_interp = interp1d(
        ta_fit_abs_data[:, 0],
        ta_fit_abs_data[:, 5],
    )
    ta_fit_PL_data, _, _ = compute_PL(
        E_vec,
        popt_PL,
        2,
        sys,
        extra_dict_params_PL,
    )

    abs_data = ta_fit_abs_interp(E_vec)

    def model_fun(xdata, *popt):
        fdepl = popt[0]
        fse = popt[1]
        """
        print(time.strftime('%X'))
        print('fdepl: %.2f, fse: %.2f' % (fdepl, fse))
        print('\n', flush=True)
        """

        return abs_data * (1 - fdepl) - fse * ta_fit_PL_data

    return model_fun


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

with open('extra/data/ta_analysis/HH/times_BS066_HHEx_1.0mW.txt') as f:
    ta_times = loadtxt(f)[:, 1]

with open('extra/extcharge/export_PL/BS066_vo_hh.csv') as f:
    ta_fit_PL_data = loadtxt(f, delimiter=',')

with open('extra/extcharge/export_abs/BS066_vo_sum.csv') as f:
    ta_fit_abs_data = loadtxt(f, delimiter=',')

time_idx = 180

p0_values = (0.1, 0.1)
bounds = ((0, 0), (1, 1))

eV_min, eV_max = ta_fit_abs_data[0, 0], ta_data[time_idx, 0, 0]
E_vec = linspace(eV_min, eV_max, 1 << 8)

ta_data = ta_data[:, ::-1, :]
ta_data_mask = (ta_data[time_idx, :, 0] > eV_min) * (ta_data[time_idx, :, 0] <
                                                     eV_max)

model_func = model(ta_fit_abs_data, ta_data[time_idx, ta_data_mask, 0])

popt, pcov = time_func(
    curve_fit,
    model_func,
    ta_data[time_idx, ta_data_mask, 0],
    ta_data[time_idx, ta_data_mask, 1],
    p0=p0_values,
    bounds=bounds,
    method='trf',
)

fdepl, fse = popt
print(time.strftime('%X'))
print('fdepl: %.4f, fse: %.4f' % (fdepl, fse))
print('\n', flush=True)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, n_files)
]

ax[0].plot(
    E_vec,
    model(ta_fit_abs_data, E_vec)(E_vec, *popt),
    color='k',
)

ax[0].plot(
    ta_fit_abs_data[:, 0],
    ta_fit_abs_data[:, 5],
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
    label='%.2f ps' % ta_times[time_idx],
)

ax[0].set_xlim(eV_min, eV_max)
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
