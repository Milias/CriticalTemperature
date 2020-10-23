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


be_bnd = 1.0

file_version = 'v1'

if file_version == 'v1':
    N_Q = 1 << 6
    Q_vec = linspace(0, 0.2, N_Q)

be_arr = zeros((N_Q))


def compute_be(be_cou_func):
    for i in range(N_Q):
        print(
            '[%d/%d] Computing %.5f' % (
                n + 1,
                N_Q,
                Q_vec[i],
            ),
            flush=True,
        )

        be_arr[i] = time_func(
            be_cou_func,
            Q_vec[i],
            1.0,
            N_k,
            sys,
            be_bnd,
        )

        save(
            'extra/data/topo/%s_data_%s' % (
                os.path.splitext(os.path.basename(__file__))[0],
                file_version,
            ),
            be_arr,
        )

        print('be: %.2f meV' % (be_arr[i, j] * 1e3))


under_threshold = 0.0

try:
    be_arr[:] = load(
        'extra/data/topo/%s_data_%s.npy' %
        (os.path.splitext(os.path.basename(__file__))[0], file_version))

    if (be_arr < under_threshold).any():
        compute_be(topo_be_b_t_eff_cou_Q, under_threshold)

except IOError as e:
    print('%s' % e, flush=True)

    compute_be(topo_be_b_t_eff_cou_Q)

print('min: %.2f meV, max: %.2f meV' % (
    1e3 * amin(be_arr[be_arr > 0]),
    1e3 * amax(be_arr[be_arr > 0]),
))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

be_arr_mask = be_arr > 0.1
be_positive_arr = be_arr[be_arr_mask]

ax[0].plot(
    X,
    Y,
    ma.masked_where(be_arr == 0, be_arr * 1e3),
    cmap=cm,
    vmin=plot_min,
    vmax=plot_max,
    shading='flat',
    #edgecolor=(1, 1, 1, 0.1),
    edgecolor=None,
    linewidth=0.005,
)

k_vec = linspace(-Q_vec[-1], Q_vec[-1], 1 << 8)
disp_vec = array([topo_eigenval_2d_v(k, sys) for k in k_vec])

ax[0].plot(
    k_vec,
    disp_vec[:, 0] / amax(abs(disp_vec[:, 0])) * Q_vec[-1],
    color='c',
)

ax[0].plot(
    k_vec,
    disp_vec[:, 2] / amax(abs(disp_vec[:, 2])) * Q_vec[-1],
    color='g',
)

ax[0].axis('equal')
ax[0].set_xlim(
    -1.1 * max(amax(X[1:, 1:][be_arr_mask]), amax(Y[1:, 1:][be_arr_mask])),
    1.1 * max(amax(X[1:, 1:][be_arr_mask]), amax(Y[1:, 1:][be_arr_mask])),
)
ax[0].set_ylim(
    -1.1 * max(amax(X[1:, 1:][be_arr_mask]), amax(Y[1:, 1:][be_arr_mask])),
    1.1 * max(amax(X[1:, 1:][be_arr_mask]), amax(Y[1:, 1:][be_arr_mask])),
)

cb = fig.colorbar(
    ScalarMappable(cmap=cm),
    ax=ax[0],
    boundaries=linspace(plot_min, plot_max, 256),
    ticks=linspace(plot_min, plot_max, n_ticks),
    pad=0.01,
    aspect=50,
)

cb.ax.set_ylabel(r'$E_X(\vec{Q})$ (meV)')

ax[0].set_xlabel(r'$Q_x$ (nm$^{-1}$)')
ax[0].set_ylabel(r'$Q_y$ (nm$^{-1}$)')

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()
