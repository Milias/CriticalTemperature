from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(2 * array([6.8, 5.3]))

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

Q_th = 0.0 * pi
Q_length = 1.0
Q_val = array([cos(Q_th), sin(Q_th)]) * Q_length
kz_val = 0.0

th_val = Q_th + 0.25 * pi
q_vec = linspace(-5.0, 5.0, 1 << 8 + 1)
result = zeros((q_vec.size, 4, 4), dtype=complex)

disp_vec = array([topo_eigenval_3d_v(k, kz_val, sys) for k in q_vec])
#disp_vec = array([topo_eigenval_2d_v(k, sys) for k in q_vec])

states_transf = array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

print(states_transf)

disp_arr = zeros((q_vec.size, 4, 4, 2))
for i, j in itertools.product(range(4), repeat=2):
    disp_arr[:, i, j, 0] = disp_vec[:, i]
    disp_arr[:, i, j, 1] = disp_vec[:, j]

for ii, q in enumerate(q_vec):
    result[ii] = array(
        topo_vert_2d_v(
            Q_val,
            [q * cos(th_val), q * sin(th_val)],
            [0.0, 0.0],
            sys,
        ),
        order='F',
    ).reshape(4, 4)

    for i in range(2):
        disp_arr[ii, :, :,
                 i] = dot(dot(
                     states_transf,
                     disp_arr[ii, :, :, i],
                 ), states_transf.T)

n_x, n_y = 4, 4
#n_x, n_y = 2, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

#band_labels = ('v', 'v\'', 'c\'', 'c')
band_labels = ('h_+', 'h_-', 'e_+', 'e_-')
band_idx_all = array(tuple(itertools.product(
    range(4),
    repeat=2,
))).reshape(4, 4, 2)

for i in range(2):
    band_idx_all[:, :, i] = dot(dot(
        states_transf,
        band_idx_all[:, :, i],
    ), states_transf.T)

for n, (i, j) in enumerate(itertools.product(range(n_x), range(n_y))):
    plot_data = result[:, i, j]

    ax[n].plot(q_vec, imag(plot_data), 'b-')
    ax[n].plot(q_vec, real(plot_data), 'r-')

    ax[n].set_xlim(q_vec[0], q_vec[-1])
    ax[n].axhline(
        y=0,
        color='k',
        linestyle='--',
        linewidth=0.7,
    )
    ax[n].axvline(
        x=0,
        color='k',
        linestyle='--',
        linewidth=0.7,
    )
    ax[n].axvline(
        x=Q_length,
        color='c',
        linestyle='-',
        linewidth=0.6,
    )
    ax[n].axvline(
        x=-Q_length,
        color='c',
        linestyle='-',
        linewidth=0.6,
    )

    ax[n].plot(
        q_vec,
        disp_arr[:, i, j, 0],
        color='m',
        linestyle='-',
        linewidth=0.6,
    )

    ax[n].plot(
        q_vec,
        disp_arr[:, i, j, 1],
        color='g',
        linestyle='-',
        linewidth=0.6,
    )

    ax[n].text(
        q_vec[0] * 0.95,
        -1.2 * 0.95,
        r'$%s,%s$' % (
            band_labels[band_idx_all[i, j, 0]],
            band_labels[band_idx_all[i, j, 1]],
        ),
        fontsize=14,
    )

    if i < n_x - 1:
        ax[n].set_xticklabels([])
    if j > 0:
        ax[n].set_yticklabels([])
    """
    if n > 0:
        ax[n].set_ylim(*ax[0].get_ylim())
    """
    ax[n].set_ylim(-1.2, 1.2)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v6'),
    transparent=True,
)

plt.show()
