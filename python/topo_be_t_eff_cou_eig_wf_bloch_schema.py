from common import *
import matplotlib.pyplot as plt
matplotlib.use('pdf')


def bloch_coords(k, theta, sys):
    return array([
        sys.params.A2 * k * cos(theta),
        -sys.params.A2 * k * sin(theta),
        sys.params.B2 * k**2 - sys.params.M,
    ]) / sqrt((sys.params.A2 * k)**2 +
              (sys.params.B2 * k**2 - sys.params.M)**2)


plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = (6.8, 1.5 * 5.3)

with open('config/topo_sys.yaml') as f:
    print('Loading "%s".' % f.name)
    settings_dict = yaml.load(f, Loader=yaml.CLoader)

globals().update(settings_dict['globals'])

params = initialize_struct(sys_params, settings_dict['params'])
sys = system_data_v2(params)

file_version = 'v1'

k_vec = array([
    -logspace(-2, 2, (1 << 8) + 1)[::-1],
    logspace(-2, 2, (1 << 8) + 1),
]).flatten()
theta_val = 0 * pi

bloch_points = bloch_coords(k_vec, theta_val, sys)

sphere_shift = 1.5
k_quiver_max = 1.0
k_quiver_plot_max = 2.0
N_quiver = 12

k_quiver_vec = linspace(-k_quiver_max, k_quiver_max, N_quiver + 1)
k_quiver_plot_vec = linspace(
    -k_quiver_plot_max,
    k_quiver_plot_max,
    k_quiver_vec.size,
)
spin_quiver = bloch_coords(k_quiver_vec, theta_val, sys)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

ax[0].quiver(
    spin_quiver[0],
    sphere_shift + spin_quiver[2],
    spin_quiver[0],
    spin_quiver[2],
    color='m',
    zorder=20,
)

ax[0].quiver(
    spin_quiver[0],
    -sphere_shift - spin_quiver[2],
    spin_quiver[0],
    -spin_quiver[2],
    color='c',
    zorder=20,
)

ax[0].quiver(
    k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    spin_quiver[0],
    spin_quiver[2],
    color='r',
    zorder=20,
)

ax[0].quiver(
    k_quiver_plot_vec,
    zeros_like(k_quiver_plot_vec),
    spin_quiver[0],
    -spin_quiver[2],
    color='b',
    zorder=20,
)

ax[0].plot(
    bloch_points[0],
    bloch_points[2] + sphere_shift,
    color='r',
    linewidth=1.6,
    zorder=30,
)

ax[0].plot(
    bloch_points[0],
    -bloch_points[2] - sphere_shift,
    color='b',
    linewidth=1.6,
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
    label='Electron',
)

ax[0].plot(
    [],
    [],
    marker='o',
    markersize=15,
    markeredgecolor='b',
    markeredgewidth=2,
    color='w',
    label='Hole',
)

for ii, k_val in enumerate(k_quiver_vec):
    ax[0].plot(
        [spin_quiver[0, ii], k_quiver_plot_vec[ii]],
        [sphere_shift + spin_quiver[2, ii], 0],
        color='r',
        linestyle='--',
        linewidth=0.6,
    )
    ax[0].plot(
        [spin_quiver[0, ii], k_quiver_plot_vec[ii]],
        [-sphere_shift - spin_quiver[2, ii], 0],
        color='b',
        linestyle='--',
        linewidth=0.6,
    )

    ax[0].plot(
        [0, spin_quiver[0, ii]],
        [sphere_shift, sphere_shift + spin_quiver[2, ii]],
        color='m',
        linestyle=':',
        linewidth=0.6,
    )
    ax[0].plot(
        [0, spin_quiver[0, ii]],
        [-sphere_shift, -sphere_shift - spin_quiver[2, ii]],
        color='c',
        linestyle=':',
        linewidth=0.6,
    )

ax[0].axhline(
    y=0,
    color='k',
    linewidth=1.1,
)

ax[0].set_frame_on(False)
ax[0].xaxis.set_visible(False)
ax[0].yaxis.set_visible(False)
ax[0].set_xlabel('$k$ (nm$^{-1}$)')
ax[0].axis('equal')

ax[0].legend(
    loc='upper right',
    prop={'size': 16},
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

#plt.show()
