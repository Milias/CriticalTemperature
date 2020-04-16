from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 1, 3
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys = system_data(m_e, m_h, eps_sol, T, size_d, 0.0, 0.0, 0.0, 0.0, eps_sol,
                  ext_dist_l)

print(sys.exc_bohr_radius_mat())
print(sys.get_E_n(0.5))

N_rho_cm = 1 << 8
N_delta_z = 4

rho_cm_vec = logspace(-2, 1, N_rho_cm) * sys.exc_bohr_radius_mat()
delta_z_vec = logspace(log10(4e-2), log10(2e-1), N_delta_z)


def wrapper(x, rho_cm, delta_z, sys):
    return exciton_dd_var_E_a0(
        *x,
        rho_cm,
        0,
        1.5 * pi,
        delta_z,
        sys,
    )


def minimize_func(rho_cm, delta_z, sys):
    r = minimize(
        wrapper,
        (0.1, sys.exc_bohr_radius_mat()),
        bounds=((0, 1), (0, None)),
        args=(rho_cm, delta_z, sys),
    )

    return (
        wrapper(r.x, rho_cm, delta_z, sys),
        *r.x,
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
)).reshape((N_rho_cm, N_delta_z, 3))

pool.terminate()

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_delta_z)
]

rho_cm_vec /= sys.exc_bohr_radius_mat()

for i in range(n_y):
    ax[i].axvline(
        x=1 / (1 + m_h / m_e),
        linewidth=0.6,
        color='m',
    )

for i in range(N_delta_z):
    ax[0].semilogx(
        rho_cm_vec,
        (sys.get_E_n(0.5) - data[:, i, 0]) * 1e3,
        linestyle='-',
        color=colors[i],
        #label=r'%s nm' % sci_notation(delta_z_vec[i]),
        label=r'$%.1f$ $\mathrm{\AA}$ ' % (delta_z_vec[i] * 10),
    )

    ax[1].semilogx(
        rho_cm_vec,
        data[:, i, 1] * 1e2,
        linestyle='-',
        color=colors[i],
    )

    ax[2].semilogx(
        rho_cm_vec,
        (data[:, i, 2] - sys.exc_bohr_radius_mat()) /
        sys.exc_bohr_radius_mat() * 1e2,
        linestyle='-',
        color=colors[i],
    )

ax[0].set_ylim(0, None)
ax[1].set_ylim(0, None)
ax[2].set_ylim(0, None)

ax[0].set_xlim(rho_cm_vec[0], rho_cm_vec[-1])
ax[1].set_xlim(rho_cm_vec[0], rho_cm_vec[-1])
ax[2].set_xlim(rho_cm_vec[0], rho_cm_vec[-1])

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])

ax[2].set_xlabel(r'$\rho_{CM} / a_0$')
ax[2].xaxis.set_label_coords(0.52, -0.07)

ax[0].set_ylabel(r'$|\Delta E_{dd}|$ (meV)')
ax[1].set_ylabel(r'$\alpha$ ($\times 10^{-2}$)')
ax[2].set_ylabel(r'$\Delta a_0/a_0$ ($\times 10^{-2}$)')

lg = ax[0].legend(
    title=r'$\Delta_z$',
    prop={'size': 12},
)
lg.get_title().set_fontsize(14)

plt.tight_layout()

fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'pot_dd_var_E_a0_min_A1',
    transparent=True,
)

plt.show()
