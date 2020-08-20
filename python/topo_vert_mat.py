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
#"""
U_mat = array(topo_orthU_3d_v(*k_val, kz_val, sys), order='F').reshape(4, 4)
U_mat_d = U_mat.T.conjugate()
eig_vals = diagflat(topo_eigenval_3d_v(sqrt(sum(k_val * k_val)), kz_val, sys))

print('U_mat')
print(U_mat)
print('U_mat_d')
print(U_mat_d)
print('U_mat_d * U_mat')
print(U_mat_d.dot(U_mat))
print()

print((U_mat_d.dot(eig_vals).dot(U_mat)).T)
hamilt = array(topo_ham_3d_v(*k_val, kz_val, sys), order='F').reshape(4, 4)

print(diag(eig_vals))
print(hamilt)
print(amax(abs(hamilt-(U_mat_d.dot(eig_vals).dot(U_mat)).T)))
print(scipy.linalg.eigvals(hamilt))

#"""

th_val = 0.25 * pi
k_vec = linspace(-0.35, 0.35, 1 << 8)
result = zeros((k_vec.size, 4, 4), dtype=complex)

disp_vec = array([topo_eigenval_3d_v(k, kz_val, sys) for k in k_vec])
#disp_vec = array([topo_eigenval_2d_v(k, sys) for k in k_vec])

for ii, k in enumerate(k_vec):
    result[ii] = array(
        topo_vert_3d_v(
            [*k_val, kz_val],
            [k * cos(th_val), k * sin(th_val), 0.0],
            sys,
        ),
        order='F',
    ).reshape(4, 4)

n_x, n_y = 4, 4
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

for n, (i, j) in enumerate(itertools.product(range(4), range(4))):
    """
    if i == 1 and j == 2:
        plot_data = -result[:, i, j]
    elif i == 2 and j == 1:
        plot_data = -result[:, i, j].conjugate()
    elif i == 3 and j == 0:
        plot_data = result[:, i, j].conjugate()
    else:
        plot_data = result[:, i, j]

    if i > j:
        plot_data = plot_data[::-1]
    """
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

    ax[n].plot(
        k_vec,
        disp_vec[:, i],
        color='m',
        linestyle='-',
        linewidth=0.6,
    )

    ax[n].plot(
        k_vec,
        disp_vec[:, j],
        color='g',
        linestyle='-',
        linewidth=0.6,
    )

    if i < 3:
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
    (os.path.splitext(os.path.basename(__file__))[0], 'v1'),
    transparent=True,
)

plt.show()
