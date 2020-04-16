from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys = system_data(m_e, m_h, eps_sol, T, size_d, 0.0, 0.0, 0.0, 0.0, eps_sol, ext_dist_l)

N_rho_cm = 1 << 8
N_delta_z = 5

rho_cm_vec = logspace(-3, 2, N_rho_cm) * sys.exc_bohr_radius_mat()
delta_z_vec = logspace(-2, log10(0.2), N_delta_z)


def minimize_func(rho_cm, delta_z, sys):
    alpha_min = minimize_scalar(
        exciton_dd_var_E,
        bracket=(0, 1),
        bounds=(0, 1),
        args=(rho_cm, 0, 1.5 * pi, delta_z, sys),
        method='bounded',
    ).x

    return (
        alpha_min,
        exciton_dd_var_E(alpha_min, rho_cm, 0, 1.5 * pi, delta_z, sys),
    )


starmap_args = [(
    rho_cm,
    delta_z,
    sys,
) for rho_cm, delta_z in itertools.product(rho_cm_vec, delta_z_vec)]

pool = multiprocessing.Pool(multiprocessing.cpu_count())

data = array(time_func(
    pool.starmap,
    minimize_func,
    starmap_args,
)).reshape((N_rho_cm, N_delta_z, 2))

pool.terminate()

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_delta_z)
]

for i in range(N_delta_z):
    ax[0].semilogx(
        rho_cm_vec / sys.exc_bohr_radius_mat(),
        (-sys.get_E_n(0.5) + data[:, i, 1]) * 1e3,
        linestyle='-',
        color=colors[i],
        #label=r'%s nm' % sci_notation(delta_z_vec[i]),
        label=r'$%.1f$ $\AA$ ' % (delta_z_vec[i] * 10),
    )

    ax[1].semilogx(
        rho_cm_vec / sys.exc_bohr_radius_mat(),
        data[:, i, 0] * 1e2,
        linestyle='-',
        color=colors[i],
    )

ax[0].set_xlim(
    rho_cm_vec[0] / sys.exc_bohr_radius_mat(),
    rho_cm_vec[-1] / sys.exc_bohr_radius_mat(),
)

ax[1].set_xlim(
    rho_cm_vec[0] / sys.exc_bohr_radius_mat(),
    rho_cm_vec[-1] / sys.exc_bohr_radius_mat(),
)
ax[1].set_ylim(0, None)

ax[0].set_xticks([])

ax[0].set_ylabel(r'$\Delta E_{dd}$ (meV)')
ax[1].set_ylabel(r'$\alpha$ ($\times 10^{-2}$)')
ax[1].set_xlabel(r'$\rho_{CM} / a_0$')

lg = ax[0].legend(
    title=r'$\Delta_z$',
    prop={'size': 12},
)
lg.get_title().set_fontsize(14)

plt.tight_layout()

fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'pot_dd_var_E_min_A1',
    transparent=True,
)

plt.show()
