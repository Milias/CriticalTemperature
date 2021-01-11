from common import *
import matplotlib.pyplot as plt
#matplotlib.use('pdf')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
})

fig_size = (6.8, 2 * 5.3)

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

file_version = 'v5'

if file_version == 'v5':
    N_Q = (1 << 7) + (1 << 6)
    Q_vec = linspace(0, 3.0, N_Q)

N_Q_max = 47
k_max = 16
print('k_max: %f' % k_max)

eig_vec = full((N_Q_max, N_k), float('nan'))

try:
    eig_vec = load('extra/data/topo/%s_data_%s.npy' % (
        os.path.splitext(os.path.basename(__file__))[0],
        file_version,
    ))

    if eig_vec.shape[0] < N_Q_max:
        eig_vec_new = full((N_Q_max, N_k), float('nan'))
        eig_vec_new[:eig_vec.shape[0]] = eig_vec
        eig_vec = eig_vec_new

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

    eig_vec[ii] = eig_arr[1]

    save(
        'extra/data/topo/%s_data_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        eig_vec,
    )

    print('[%d/%d(%d)] (Q, E_Q): (%f, %f)' % (
        ii + 1,
        N_Q_max,
        N_Q,
        Q_val,
        eig_arr[0, 0],
    ))

    del eig_arr

phi_val = 0.0 * pi
k_vec = linspace(0, k_max, N_k)


def compute_bloch_vec(Qk, phi):
    Q, k = Qk
    return topo_bloch_th(Q, phi, k, sys)


pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

bloch_vec = array(
    time_func(
        pool.map,
        functools.partial(compute_bloch_vec, phi=phi_val),
        itertools.product(Q_vec[:N_Q_max], k_vec),
    )).reshape((N_Q_max, N_k, 4, 3))

pool.close()

bloch_vec *= repeat(
    repeat(
        eig_vec.reshape((N_Q_max, N_k, 1, 1))**2,
        repeats=4,
        axis=2,
    ),
    repeats=3,
    axis=3,
)

bloch_points = trapz(bloch_vec, k_vec, axis=1).transpose((0, 2, 1))

k_vec = array([-Q_vec[::-1], Q_vec]).flatten()

sphere_shift = 0
N_quiver = 12
k_quiver_max = Q_vec[N_Q_max - 1]
k_quiver_plot_max = amax(bloch_points[:, 0]) * 1.1

k_quiver_vec = linspace(Q_vec[0], k_quiver_max, N_quiver)
k_quiver_plot_vec = linspace(0, k_quiver_plot_max, N_quiver)
spin_quiver = bloch_points[linspace(0, N_Q_max - 1, N_quiver, dtype=int)]

idx_state_1 = 0
idx_state_2 = 2

band_labels = ('h_+', 'h_-', 'e_+', 'e_-')

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

ax[0].quiver(
    -spin_quiver[:, 0, idx_state_1],
    sphere_shift + spin_quiver[:, 2, idx_state_1],
    -spin_quiver[:, 0, idx_state_1],
    spin_quiver[:, 2, idx_state_1],
    color='m',
    zorder=20,
)

ax[0].quiver(
    spin_quiver[:, 0, idx_state_1],
    sphere_shift + spin_quiver[:, 2, idx_state_1],
    spin_quiver[:, 0, idx_state_1],
    spin_quiver[:, 2, idx_state_1],
    color='m',
    zorder=20,
)

ax[0].quiver(
    -spin_quiver[:, 0, idx_state_2],
    -sphere_shift + spin_quiver[:, 2, idx_state_2],
    -spin_quiver[:, 0, idx_state_2],
    spin_quiver[:, 2, idx_state_2],
    color='c',
    zorder=20,
)

ax[0].quiver(
    spin_quiver[:, 0, idx_state_2],
    -sphere_shift + spin_quiver[:, 2, idx_state_2],
    spin_quiver[:, 0, idx_state_2],
    spin_quiver[:, 2, idx_state_2],
    color='c',
    zorder=20,
)

ax[0].quiver(
    -k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    -spin_quiver[:, 0, idx_state_1],
    spin_quiver[:, 2, idx_state_1],
    color='r',
    zorder=20,
)

ax[0].quiver(
    k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    spin_quiver[:, 0, idx_state_1],
    spin_quiver[:, 2, idx_state_1],
    color='r',
    zorder=20,
)

ax[0].quiver(
    -k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    -spin_quiver[:, 0, idx_state_2],
    spin_quiver[:, 2, idx_state_2],
    color='b',
    zorder=20,
)

ax[0].quiver(
    k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    spin_quiver[:, 0, idx_state_2],
    spin_quiver[:, 2, idx_state_2],
    color='b',
    zorder=20,
)

ax[0].plot(
    -bloch_points[:, 0, idx_state_1],
    bloch_points[:, 2, idx_state_1] + sphere_shift,
    color='r',
    linewidth=1.6,
    zorder=30,
)

ax[0].plot(
    bloch_points[:, 0, idx_state_1],
    bloch_points[:, 2, idx_state_1] + sphere_shift,
    color='r',
    linewidth=1.6,
    zorder=30,
)

ax[0].plot(
    -bloch_points[:, 0, idx_state_2],
    bloch_points[:, 2, idx_state_2] - sphere_shift,
    color='b',
    linewidth=1.6,
    zorder=30,
)

ax[0].plot(
    bloch_points[:, 0, idx_state_2],
    bloch_points[:, 2, idx_state_2] - sphere_shift,
    color='b',
    linewidth=1.6,
    zorder=30,
)

ax[0].plot(
    -k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    color='w',
    linestyle='',
    marker='o',
    markeredgewidth=0.8,
    markeredgecolor='k',
    markersize=5,
    zorder=30,
)

ax[0].plot(
    k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    color='w',
    linestyle='',
    marker='o',
    markeredgewidth=0.8,
    markeredgecolor='k',
    markersize=5,
    zorder=30,
)

ax[0].plot(
    [],
    [],
    marker='o',
    markersize=15,
    markeredgecolor='r',
    markeredgewidth=2,
    color='w',
    label=
    r'$\left\langle \boldsymbol{\hat\Gamma}_{%s}(\boldsymbol{Q}) \right\rangle$'
    % band_labels[idx_state_1],
)

ax[0].plot(
    [],
    [],
    marker='o',
    markersize=15,
    markeredgecolor='b',
    markeredgewidth=2,
    color='w',
    label=
    r'$\left\langle \boldsymbol{\hat\Gamma}_{%s}(\boldsymbol{Q}) \right\rangle$'
    % band_labels[idx_state_2],
)

for ii, k_val in enumerate(k_quiver_vec):
    ax[0].plot(
        [-spin_quiver[ii, 0, idx_state_1], -k_quiver_plot_vec[ii]],
        [sphere_shift + spin_quiver[ii, 2, idx_state_1], idx_state_1],
        color='r',
        linestyle='--',
        linewidth=0.6,
    )
    ax[0].plot(
        [spin_quiver[ii, 0, idx_state_1], k_quiver_plot_vec[ii]],
        [sphere_shift + spin_quiver[ii, 2, idx_state_1], idx_state_1],
        color='r',
        linestyle='--',
        linewidth=0.6,
    )
    ax[0].plot(
        [-spin_quiver[ii, 0, idx_state_2], -k_quiver_plot_vec[ii]],
        [-sphere_shift + spin_quiver[ii, 2, idx_state_2], idx_state_1],
        color='b',
        linestyle='--',
        linewidth=0.6,
    )
    ax[0].plot(
        [spin_quiver[ii, 0, idx_state_2], k_quiver_plot_vec[ii]],
        [-sphere_shift + spin_quiver[ii, 2, idx_state_2], idx_state_1],
        color='b',
        linestyle='--',
        linewidth=0.6,
    )

    ax[0].plot(
        [0, -spin_quiver[ii, 0, idx_state_1]],
        [sphere_shift, sphere_shift + spin_quiver[ii, 2, idx_state_1]],
        color='m',
        linestyle=':',
        linewidth=0.6,
    )
    ax[0].plot(
        [0, spin_quiver[ii, 0, idx_state_1]],
        [sphere_shift, sphere_shift + spin_quiver[ii, 2, idx_state_1]],
        color='m',
        linestyle=':',
        linewidth=0.6,
    )
    ax[0].plot(
        [0, -spin_quiver[ii, 0, idx_state_2]],
        [-sphere_shift, -sphere_shift + spin_quiver[ii, 2, idx_state_2]],
        color='c',
        linestyle=':',
        linewidth=0.6,
    )
    ax[0].plot(
        [0, spin_quiver[ii, 0, idx_state_2]],
        [-sphere_shift, -sphere_shift + spin_quiver[ii, 2, idx_state_2]],
        color='c',
        linestyle=':',
        linewidth=0.6,
    )

ax[0].axhline(
    y=0,
    color='k',
    linewidth=1.1,
)

#ax[0].set_frame_on(False)
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_visible(False)
ax[0].set_ylim(-1, 1)

ax[0].legend(
    loc='upper center',
    prop={'size': 18},
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()
