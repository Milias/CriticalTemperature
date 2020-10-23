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

Q_val = 0.1

N_k = 1 << 6
k_vec = linspace(0, 1, N_k)

result = zeros((4, 4, N_k, N_k), dtype=complex)

for i, j in itertools.product(range(4), repeat=2):
    print((i, j), flush=True)
    result[i, j] = array(time_func(
        topo_eff_cou_Q_ij_mat,
        Q_val,
        i,
        j,
        N_k,
        sys,
    )).reshape(N_k, N_k)[::-1, ::-1]

    print("", flush=True)

print(amax(real(result)))
print(amin(real(result)))
print(average(real(result)))

print(amax(imag(result)))
print(amin(imag(result)))
print(average(imag(result)))

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

plot_max = 2

for n, (i, j) in enumerate(itertools.product(range(n_x), range(n_y))):
    ax[n].imshow(
        real(result[i, j]),
        cmap=cm.cividis,
        aspect='auto',
        extent=(0, 1, 0, 1),
        interpolation='none',
        vmin=-plot_max,
        vmax=plot_max,
        origin='lower',
    )

    ax[n].plot(
        [0.5],
        [0.95],
        color='w',
        marker=r'$%s%s \rightarrow %s%s$' % (
            band_labels[band_idx_all[i, j, 1]],
            band_labels[band_idx_all[i, j, 3]],
            band_labels[band_idx_all[i, j, 0]],
            band_labels[band_idx_all[i, j, 2]],
        ),
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
