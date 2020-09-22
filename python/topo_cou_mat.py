from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8 * 2, 5.3 * 2]))


def compute_cou_mat(result, q_vec, q_ang, k_vec, sys):
    for ii, q in enumerate(q_vec):
        result[ii] = array(
            topo_cou_2d_v(
                k_vec,
                -k_vec,
                [q * cos(q_ang), q * sin(q_ang)],
                sys,
            ),
            order='F',
        ).reshape(16, 16)


states_transf = array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
])

cou_transf = kron(
    eye(2),
    kron(
        array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]),
        eye(2),
    ),
)

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

N_th = 9

q_ang_vec = linspace(0, 2, N_th) * pi
k_vec = array([0.05, 0])
q_vec = linspace(-0.35, 0.35, 1 << 10)

result = zeros((N_th, q_vec.size, 16, 16), dtype=complex)

for ii, q_ang in enumerate(q_ang_vec):
    compute_cou_mat(
        result[ii],
        q_vec,
        arctan2(k_vec[1], k_vec[0]) + q_ang,
        k_vec,
        sys,
    )

#n_x, n_y = 16, 16
n_x, n_y = 4, 4
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

band_labels = ('v', 'v\'', 'c\'', 'c')
states_transf_16 = kron(states_transf, states_transf)
band_idx_all = array(tuple(itertools.product(
    range(4),
    repeat=4,
))).reshape(4, 4, 4, 4, 4)
band_idx_all = band_idx_all.transpose(0, 2, 1, 3, 4).reshape(16, 16, 4)

for i in range(4):
    band_idx_all[:, :, i] = dot(dot(
        states_transf_16,
        band_idx_all[:, :, i],
    ), states_transf_16.T)
    band_idx_all[:, :, i] = dot(dot(
        cou_transf,
        band_idx_all[:, :, i],
    ), cou_transf.T)

plot_max = 14

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_th)
]

for n, (i, j) in enumerate(itertools.product(range(n_x), range(n_y))):
    for ii in range(N_th):
        ax[n].plot(
            q_vec,
            real(result[ii, :, i, j]),
            linestyle='-',
            linewidth=0.9,
            color=colors[ii],
            label=r'$\theta: %.2f\pi$' % (q_ang_vec[ii] / pi),
        )
        ax[n].plot(
            q_vec,
            imag(result[ii, :, i, j]),
            linestyle='--',
            linewidth=0.9,
            color=colors[ii],
        )

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

    ax[n].text(
        q_vec[0] * 0.98,
        -plot_max * 0.98,
        r'$%s%s \rightarrow %s%s$' % (
            band_labels[band_idx_all[i, j, 1]],
            band_labels[band_idx_all[i, j, 3]],
            band_labels[band_idx_all[i, j, 0]],
            band_labels[band_idx_all[i, j, 2]],
        ),
        fontsize=14,
    )

    if i < n_x - 1:
        ax[n].set_xticklabels([])
    if j > 0:
        ax[n].set_yticklabels([])

    ax[n].set_ylim(-plot_max, plot_max)

ax[0].legend(
    loc='upper left',
    prop={'size': 9},
)
plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()
