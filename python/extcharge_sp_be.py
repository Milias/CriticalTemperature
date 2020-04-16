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

N_rho_cm = 3

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_h, T = 0.27, 0.45, 294  # K

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)


def sp_be_func(a, dz, qz, rho_cm, sys):
    return sys.get_E_n(0.5) / (1 + 8 * a**2) + 9 * a * sqrt(3 - 3 * a**2) * (
        16 * a**2 -
        7) / (32 * (1 + 8 * a**2)) * sys.c_hbarc**2 * qz * dz * rho_cm / (
            sys.m_p * (qz**2 + rho_cm**2)**2.5)


colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_rho_cm)
]


def calc_data(rho, sys, n_rho):

    return


rho_cm_vec = array([0.5, 1, 4]) * sys_sol.exc_bohr_radius_mat()
a_vec = linspace(0, 1, 1 << 8)

data = array([
    sp_be_func(a_vec, 1e-1, size_d * 0.5, rho_cm, sys_sol)
    for rho_cm in rho_cm_vec
])

for n_rho_cm in range(N_rho_cm):
    ax[0].plot(
        a_vec,
        data[:, n_rho_cm],
        color=colors[n_rho_cm],
        linestyle='-',
        linewidth=1.5,
        label=r'%.1f' % (rho_cm_vec[n_rho_cm] / sys_sol.exc_bohr_radius_mat()),
    )

ax[0].set_xlim(0, 1)
#ax[0].set_ylim(-3.5, 0.5)

ax[0].set_xticklabels(ax[0].get_xticklabels()[:-1] + [''])

ax[0].set_ylabel(r'$V(\theta)$ (eV)')
lg = ax[1].legend(
    loc='lower center',
    title=r'$|\vec{\rho}_e - \vec{\rho}_h|~/~a_0$',
    prop={'size': 12},
)
lg.get_title().set_fontsize(13)

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'sp_be_A1',
    transparent=True,
)

plt.show()
