from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8 * 2, 5.3 * 2]))

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

k_val = array([0.08, 0.03])
kz_val = 0.00

th_val = 0.25 * pi
k_vec = linspace(-0.35, 0.35, 1 << 8)
result = zeros((k_vec.size, 16, 16), dtype=complex)

for ii, k in enumerate(k_vec):
    result[ii] = array(
        topo_cou_3d_v(
            [*k_val, kz_val],
            [*(-k_val), -kz_val],
            [k * cos(th_val), k * sin(th_val), 0.0],
            sys,
        ),
        order='F',
    ).reshape(16, 16)

n_x, n_y = 16, 16
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

band_labels = ('v', 'v\'', 'c', 'c\'')
band_labels_pairs = tuple(itertools.product(band_labels, repeat=2))
band_labels_all = []

for p1 in band_labels_pairs:
    for p2 in band_labels_pairs:
        band_labels_all.append((*p1, *p2))

print(band_labels_all)

for n, (i, j) in enumerate(itertools.product(range(n_x), range(n_y))):
    plot_data = result[:, i, j]

    ax[n].plot(k_vec, real(plot_data), 'r-')
    ax[n].plot(k_vec, imag(plot_data), 'b-')

    ax[n].set_xlim(k_vec[0], k_vec[-1])
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
        k_vec[0],
        -13,
        '%s%s,%s%s' % band_labels_all[n],
        fontsize=10,
    )

    if i < 15:
        ax[n].set_xticklabels([])
    if j > 0:
        ax[n].set_yticklabels([])

    ax[n].set_ylim(-13, 13)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()
