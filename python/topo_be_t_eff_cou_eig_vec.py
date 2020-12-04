from common import *
import matplotlib.pyplot as plt
matplotlib.use('pdf')

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = (6.8, 5.3)

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
N_Q_max = 39
k_max = 8.0
print('k_max: %f' % k_max)

k_vec = linspace(-Q_vec[-1], Q_vec[-1], 1 << 8)
k_int_vec = linspace(0.0, Q_vec[-1], 1 << 8)
disp_vec = array([topo_eigenval_2d_v(0.5 * k, sys) for k in k_vec])
disp_int_vec = array(
    [amin([topo_disp_t_int(Q, k, sys) for k in k_int_vec]) for Q in Q_vec])

eig_val = full((N_Q_max, N_k), float('nan'))
eig_vec = full((N_Q_max, N_k, N_k), float('nan'))

try:
    loaded_data = load(
        'extra/data/topo/%s_data_%s.npy' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version))

    eig_val[:loaded_data.shape[0]] = loaded_data

except IOError as e:
    print('%s' % e, flush=True)

for ii, Q_val in enumerate(Q_vec[:N_Q_max]):
    if not isnan(eig_val[ii, 0]):
        continue

    eig_arr = array(time_func(
        topo_t_eff_cou_eig,
        Q_val,
        k_max,
        N_k,
        sys,
    )).reshape(N_k + 1, N_k)

    eig_val[ii] = eig_arr[0]
    eig_vec[ii] = eig_arr[1:]

    save(
        'extra/data/topo/%s_data_%s' % (
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

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_states_max)
]

plot_vec = array([
    flatnonzero(eig_val[n_q] < disp_int_vec[N_Q_max])[:N_states_max].size
    for n_q in range(N_Q_max)
])

plot_data = full((N_Q, N_states_max), float('nan'))

for ii, n_s_max in enumerate(plot_vec):
    plot_data[:N_Q_max, :n_s_max] = eig_val[:, arange(n_s_max)]

for ii in range(N_states_max - 1, -1, -1):
    kwargs = {
        'color': colors[ii],
        'linestyle': '-',
        'linewidth': 0.7,
    }

    ax[0].plot(
        Q_vec,
        plot_data[:, ii],
        **kwargs,
    )

    kwargs['label'] = '$E_{%d}$' % (ii + 1)

    ax[0].plot(
        -Q_vec,
        plot_data[:, ii],
        **kwargs,
    )

ax[0].axhline(
    y=0,
    color='k',
    linestyle='-',
    linewidth=0.6,
)

ax[-1].plot(
    Q_vec,
    disp_int_vec,
    color='b',
    linewidth=0.7,
)

ax[0].plot(
    k_vec,
    disp_vec[:, 2],
    color='g',
    linewidth=0.9,
)

ax[0].plot(
    k_vec,
    disp_vec[:, 0],
    color='c',
    linewidth=0.9,
)

ax[0].plot(
    -Q_vec,
    disp_int_vec,
    color='b',
    linewidth=0.9,
)

ax[0].set_xlim(-Q_vec[-1], Q_vec[-1])

ax[0].set_xlabel('Q (nm$^{-1}$)')
ax[0].set_ylabel('E (eV)')

ax[0].legend(
    loc='center right',
    prop={'size': 10},
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

#plt.show()
