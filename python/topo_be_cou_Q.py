from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([2 * 6.8, 2 * 5.3]))

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)


def find_eps_sol(N_k, sys, be_cou, be_bnd):
    def find_f(eps_sol):
        sys.params.eps_sol = eps_sol
        return topo_be_p_cou(N_k, sys, be_bnd) - be_cou

    return root_scalar(
        find_f,
        bracket=(2, 10),
        method='brentq',
    ).root


be_bnd_positive = 0.3
be_bnd_negative = 0.35

file_version = 'v4'

if file_version == 'v2':
    N_Q = 1 << 6
    N_Q_th = 1 << 6
    Q_vec = linspace(0, 0.5, N_Q)
    Q_th_vec = linspace(0, 2, N_Q_th) * pi
elif file_version == 'v3':
    N_Q = 1 << 7
    N_Q_th = 1 << 7
    Q_vec = linspace(0, 0.15, N_Q)
    Q_th_vec = linspace(0, 2, N_Q_th) * pi
elif file_version == 'v4':
    N_Q = 1 << 6
    N_Q_th = (1 << 6) + 1
    Q_vec = linspace(0, 0.15, N_Q)
    Q_th_vec = linspace(0, 1, N_Q_th) * pi

be_arr = zeros((N_Q, N_Q_th))

try:
    be_arr[:] = load(
        'extra/data/topo/%s_data_%s.npy' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version))
except IOError as e:
    print('%s' % e, flush=True)
    be_arr[0, :] = time_func(topo_be_t_eff_cou, 1.0, N_k, sys, be_bnd_positive)

    save(
        'extra/data/topo/%s_data_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        be_arr,
    )

    for n, (i, j) in enumerate(itertools.product(range(1, N_Q),
                                                 range(N_Q_th))):
        print(
            '[%d/%d] Computing (%.3f, %.3f)' % (
                n + 1,
                (N_Q - 1) * N_Q_th,
                *tuple(Q_vec[i] * array([cos(Q_th_vec[j]),
                                         sin(Q_th_vec[j])])),
            ),
            flush=True,
        )

        save(
            'extra/data/topo/%s_data_%s' % (
                os.path.splitext(os.path.basename(__file__))[0],
                file_version,
            ),
            be_arr,
        )

        be_arr[i, j] = time_func(
            topo_be_t_eff_cou_Q,
            Q_vec[i] *
            array([cos(Q_th_vec[j]), sin(Q_th_vec[j])]),
            1.0,
            N_k,
            sys,
            be_bnd_positive if Q_th_vec[j] < 0.5 * pi else be_bnd_negative,
        )

    save(
        'extra/data/topo/%s_data_%s' % (
            os.path.splitext(os.path.basename(__file__))[0],
            file_version,
        ),
        be_arr,
    )

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, 1 << 8)
]

cm = ListedColormap(colors)

if Q_th_vec[-1] < 2 * pi:
    Q, Ph = meshgrid(
        linspace(0, Q_vec[-1], N_Q + 1),
        linspace(0, 2, 2 * N_Q_th) * pi,
        indexing='ij',
    )
    X, Y = Q * cos(Ph), Q * sin(Ph)

    be_arr_old = be_arr[:]
    be_arr = zeros((N_Q, 2 * N_Q_th - 1))
    be_arr[:, :N_Q_th] = be_arr_old[:]
    be_arr[:, N_Q_th:] = be_arr_old[:, -2::-1]
else:
    Q, Ph = meshgrid(
        linspace(0, Q_vec[-1], N_Q + 1),
        linspace(0, 2, N_Q_th) * pi,
        indexing='ij',
    )
    X, Y = Q * cos(Ph), Q * sin(Ph)

be_arr_mask = be_arr > 0.1
be_positive_arr = be_arr[be_arr_mask]

plot_min, plot_max = 0.15, 0.35

ax[0].pcolormesh(
    X,
    Y,
    ma.masked_where(be_arr == 0, be_arr),
    cmap=cm,
    vmin=plot_min,
    vmax=plot_max,
    shading='flat',
    edgecolor=(1, 1, 1, 0.1),
    #edgecolor=None,
    linewidth=0.005,
)

ax[0].axis('equal')
ax[0].set_xlim(
    1.2 * min(amin(X[:-1, :-1][be_arr_mask]), amin(Y[:-1, :-1][be_arr_mask])),
    1.2 * min(amax(X[:-1, :-1][be_arr_mask]), amax(Y[:-1, :-1][be_arr_mask])),
)
ax[0].set_ylim(
    1.2 * max(amin(X[:-1, :-1][be_arr_mask]), amin(Y[:-1, :-1][be_arr_mask])),
    1.2 * max(amax(X[:-1, :-1][be_arr_mask]), amax(Y[:-1, :-1][be_arr_mask])),
)

cb = fig.colorbar(
    ScalarMappable(cmap=cm),
    ax=ax[0],
    boundaries=linspace(plot_min, plot_max, 256),
    ticks=linspace(plot_min, plot_max, 9),
    pad=0.01,
    aspect=50,
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()
