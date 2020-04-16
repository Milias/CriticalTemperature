from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0
delta_z = 0.4

sys = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol, ext_dist_l)

N_rho_cm = 5
N_alpha = 1 << 8

rho_cm_vec = logspace(-1, 1, N_rho_cm) * sys.exc_bohr_radius_mat()
alpha_vec = linspace(0, 0.5, N_alpha)

pool = multiprocessing.Pool(multiprocessing.cpu_count())

starmap_args = [(
    alpha,
    rho_cm,
    0,
    1.5 * pi,
    delta_z,
    sys,
) for alpha, rho_cm in itertools.product(alpha_vec, rho_cm_vec)]

data = array(time_func(
    pool.starmap,
    exciton_dd_var_E,
    starmap_args,
)).reshape((N_alpha, N_rho_cm))

pool.terminate()

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_rho_cm)
]

for i in range(N_rho_cm):
    alpha_min = minimize_scalar(
        exciton_dd_var_E,
        bracket=(0, 1),
        bounds=(0, 1),
        args=(rho_cm_vec[i], 0, 1.5 * pi, delta_z, sys),
        method='bounded',
    )

    print(alpha_min)

    ax[0].plot(
        alpha_vec,
        data[:, i],
        color=colors[i],
        linestyle='-',
        label='%.1f' % (rho_cm_vec[i] / sys.exc_bohr_radius_mat()),
    )

    ax[0].axvline(x=alpha_min.x, linewidth=0.6, color=colors[i])

ax[0].set_xlim(alpha_vec[0], alpha_vec[-1])

ax[0].set_xlabel(r'$\alpha$')
ax[0].set_ylabel(r'$E$ (eV)')

lg = ax[0].legend(
    title=r'$\rho_{CM} / a_0$',
    prop={'size': 12},
)
lg.get_title().set_fontsize(14)

plt.tight_layout()

fig.subplots_adjust(wspace=0.01, hspace=0.01)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'pot_dd_var_E_A1',
    transparent=True,
)

plt.show()
