from common import *
import matplotlib.pyplot as plt

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


def find_eps_sol(N_k, sys, be_cou, be_bnd):
    def find_f(eps_sol):
        sys.params.eps_sol = eps_sol
        return topo_be_p_cou(N_k, sys, be_bnd) - be_cou

    return root_scalar(
        find_f,
        bracket=(2, 10),
        method='brentq',
    ).root


"""
eps_sol = find_eps_sol(N_k, sys, 193e-3, 1)
print(eps_sol)
"""

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

Q_val = 25.0
print('Q: %f' % Q_val)

k_vec = linspace(1 / N_k, Q_val, N_k)

eig_arr = array(topo_p_cou_eig(Q_val, N_k, sys)).reshape(N_k + 1, N_k)
eig_val = eig_arr[0]
eig_vec = eig_arr[1:]

#print(-2 * sys.c_aEM**2 / sys.params.eps_sol**2 * sys.d_params.m_p)

plot_Q_val = 20.0
k_plot_vec = linspace(0, plot_Q_val, 1 << 10)
disp_int_plot_vec = array([topo_disp_p_int(0, k, sys) for k in k_plot_vec])

k_vec = linspace(0, Q_val, N_k)
disp_int_vec = array([topo_disp_p_int(0, k, sys) for k in k_vec])

print('disp_min: %f eV' % amin(disp_int_vec))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

plot_vec = flatnonzero(eig_val < amin(disp_int_vec))
plot_eig_val = eig_val[plot_vec]

print(plot_eig_val)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, plot_vec.size)
]

for ii, val in enumerate(plot_eig_val):
    ax[0].plot(
        [k_vec[ii]],
        [val],
        linestyle='',
        marker='o',
        markersize=4,
        markeredgecolor='k',
        markeredgewidth=0.5,
        color=colors[ii],
        zorder=100,
    )
"""
ax[0].plot(
    k_vec[amax(plot_vec):],
    eig_val[amax(plot_vec):],
    marker='.',
    markersize=2,
    linestyle='-',
    linewidth=1.1,
    color='m',
)
"""

ax[0].plot(
    k_plot_vec,
    disp_int_plot_vec,
    linestyle='-',
    linewidth=0.9,
    color='m',
)

for eig_idx in plot_vec:
    wf_interp = interp1d(
        k_vec,
        eig_vec[eig_idx],
        kind='cubic',
        copy=False,
        assume_sorted=True,
    )

    ax[0].plot(
        k_plot_vec,
        wf_interp(k_plot_vec),
        color=colors[eig_idx],
        linestyle='-',
        linewidth=1.0,
        label='$E_{%d} = %.5f$ eV' % (eig_idx + 1, eig_val[eig_idx]),
        zorder=-eig_idx,
    )

    """
    ax[0].plot(
        k_vec,
        eig_vec[eig_idx],
        color='w',
        linestyle='',
        marker='.',
        markersize=4,
        markeredgewidth=0.4,
        markeredgecolor=colors[eig_idx],
        zorder=-eig_idx,
    )
    """

ax[0].axhline(
    y=0,
    color='k',
    linewidth=0.7,
)

ax[0].set_xlim(0, plot_Q_val)
ax[0].set_ylim(
    1.1 * amin(eig_vec[plot_vec, :]),
    2 * eig_vec[0, 0],
)

ax[0].legend(
    loc='upper right',
    prop={'size': 11},
    title='Bound state energy',
)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/TopoExciton/%s_%s.pdf' %
    (os.path.splitext(os.path.basename(__file__))[0], file_version),
    transparent=True,
)

plt.show()