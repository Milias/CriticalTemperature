from common import *
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8 * 2, 5.3 * 2]))

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

N_alpha = 1
k_vec = linspace(0, 1, N_k)
alpha_vec = linspace(-1, 1, N_alpha)
result = zeros((1, N_k, N_k), dtype=complex)
plot_result = zeros((N_alpha, N_k, N_k), dtype=complex)

for ii, alpha in enumerate([1]):
    print('[%d/%d] ⍺: %.3f' % (ii + 1, result.shape[0], alpha), flush=True)
    result[ii] = array(time_func(
        topo_cou_mat,
        N_k,
        sys,
    )).reshape(N_k, N_k)[::-1, ::-1]

    print("", flush=True)


def plot_result_func(alpha):
    return result[0] * alpha


pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
plot_result[:] = time_func(pool.map, plot_result_func, alpha_vec)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

if len(ax) != N_alpha:
    print('N_alpha != len(ax)')
    exit()

plot_max = 15

for n in range(N_alpha):
    ax[n].imshow(
        clip(real(plot_result[n]), -plot_max, plot_max),
        cmap=cm.cividis,
        aspect='auto',
        extent=(1 / N_k, 1, 1 / N_k, 1),
        interpolation='none',
        vmin=-plot_max,
        vmax=plot_max,
        origin='lower',
    )

    ax[n].plot(
        [0.5],
        [0.95],
        color='w',
        marker=r'$\alpha$: $%.3f$' % alpha_vec[n],
        markeredgecolor='k',
        markeredgewidth=0.2,
        markersize=100,
    )

    ax[n].set_xticklabels([])
    ax[n].set_yticklabels([])

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v3'),
    transparent=True,
)

plt.show()
