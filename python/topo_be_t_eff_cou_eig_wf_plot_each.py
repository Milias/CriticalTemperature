from common import *


def plot_wf(eig_idx, eig_vec, eig_val, Q_val):
    import matplotlib.pyplot as plt
    matplotlib.use('pdf')

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern'],
        'text.usetex': False,
    })

    fig_size = (6.8, 5.3)

    n_x, n_y = 1, 1
    fig = plt.figure(figsize=fig_size)
    ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

    plot_k_max = 0.5
    k_plot_vec = linspace(0, plot_k_max, 1 << 10)
    disp_int_plot_vec = array([topo_disp_t_int(Q_val, k, sys) for k in k_plot_vec])

    # 0 and 8 come from 1 / N_k and k_max.
    k_vec = linspace(0, 8, eig_vec[eig_idx].size)

    colors = [
        matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
        for h in linspace(0, 0.7, eig_vec.shape[0])
    ]

    ax[0].plot(
        k_plot_vec,
        disp_int_plot_vec,
        linestyle='-',
        linewidth=0.7,
        color='m',
        zorder=-101,
    )

    wf_interp = interp1d(
        k_vec,
        eig_vec[eig_idx],
        kind='cubic',
        copy=False,
        assume_sorted=True,
    )

    ax[0].plot(
        k_plot_vec,
        wf_interp(k_plot_vec),
        color=colors[eig_idx],
        linestyle='-',
        linewidth=1.0,
        label='$\phantom{\Delta} E_{%d} = %.3f$ eV\n$\Delta_{%d} = %.1f$ meV' %
        (
            eig_idx + 1,
            eig_val[eig_idx],
            eig_idx + 1,
            1e3 * (amin(disp_int_vec) - eig_val[eig_idx]),
        ),
        zorder=-eig_idx,
    )

    ax[0].plot(
        k_vec,
        eig_vec[eig_idx],
        color='w',
        linestyle='',
        marker='.',
        markersize=4,
        markeredgewidth=0.4,
        markeredgecolor=colors[eig_idx],
        zorder=-eig_idx,
    )

    ax[0].set_xlabel('k (nm$^{-1}$)')

    ax[0].set_xlim(0, plot_k_max)
    ax[0].set_ylim(0, 7)

    ax[0].legend(
        loc='upper right',
        prop={'size': 12},
        title='Bound state energy\n$Q = %.5f$ nm$^{-1}$' % Q_val,
    )

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(
        '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.png' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version),
        transparent=True,
    )


with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

file_version = 'v3'

if file_version == 'v1':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.2, N_Q)
elif file_version == 'v2':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.045, N_Q)
elif file_version == 'v3':
    N_Q = 1 << 7
    Q_vec = linspace(0, 0.125, N_Q)

N_states_max = 6
N_Q_max = 10
k_max = 8.0
print('k_max: %f' % k_max)

eig_vec = full((N_states_max, N_Q_max, N_k), float('nan'))
eig_val = full((N_states_max, N_Q_max), float('nan'))

try:
    eig_vec = load(
        'extra/data/topo/%s_eig_vec_%s.npy' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version))
    eig_val = load(
        'extra/data/topo/%s_eig_val_%s.npy' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version))

except IOError as e:
    print('%s' % e, flush=True)

for ii, Q_val in enumerate(Q_vec[:N_Q_max]):
    if not isnan(eig_vec[ii, 0]):
        continue

    eig_arr = array(time_func(
        topo_t_eff_cou_eig,
        Q_val,
        k_max,
        N_k,
        sys,
    )).reshape(N_k + 1, N_k)

    eig_val[ii] = eig_arr[0, :N_states_max]
    eig_vec[ii] = eig_arr[1:N_states_max + 1]

    save(
        'extra/data/topo/%s_eig_vec_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        eig_vec,
    )

    save(
        'extra/data/topo/%s_eig_val_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        eig_val,
    )

    print('[%d/%d] (Q, E_Q): (%f, %f)' % (
        ii + 1,
        N_Q,
        Q_val,
        eig_arr[0, 0],
    ))

    del eig_arr
