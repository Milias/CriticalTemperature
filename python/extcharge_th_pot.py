from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 3, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_y, n_x, i + 1) for i in range(n_x * n_y)]


def ke_k_pot(u, x, sys):
    pot = (sys.eps_mat + sys.eps_r * tanh(u * sys.size_d / x * 0.5)) / (
        sys.eps_r + sys.eps_mat * tanh(u * sys.size_d / x * 0.5))

    return -special.j0(u) * pot / (x * sys.eps_mat) * sys.c_aEM * sys.c_hbarc


def ext_k_pot(u, x, sys):
    pot = 2 * exp(-u * (sys.ext_dist_l + 0.5 * sys.size_d) /
                  x) / (sys.eps_mat + sys.eps_r +
                        (sys.eps_r - sys.eps_mat) * exp(-u * sys.size_d / x))

    return -special.j0(u) * pot / x * sys.c_aEM * sys.c_hbarc


N_rho_cm = 3
N_a0 = 7
N_th = 128

size_d = 1  # nm
eps_sol = 1
eps_mat = 6 * eps_sol
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_mat = system_data(m_e, m_h, eps_sol, T, size_d, eps_mat, ext_dist_l)

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_a0)
]


def integ_pot(pot_func, x, sys):
    return sum([
        quad(
            pot_func,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=1000,
            args=(x, sys),
        )[0] for n in arange(1e4)
    ])


def calc_data(rho_cm, a0, th, sys, n_a0, ke_vec):
    rho_e = sqrt(rho_cm**2 + a0**2 / (1 + sys.m_eh)**2 +
                 2 * rho_cm * a0 * cos(th) / (1 + sys.m_eh))
    rho_h = sqrt(rho_cm**2 + a0**2 / (1 + 1 / sys.m_eh)**2 -
                 2 * rho_cm * a0 * cos(th) / (1 + 1 / sys.m_eh))

    return ke_vec[n_a0] + integ_pot(ext_k_pot, rho_e, sys) - integ_pot(
        ext_k_pot, rho_h, sys)


rho_cm_vec = array([0.1, 1, 10]) * sys_mat.exc_bohr_radius_mat()
a0_vec = logspace(-1, 1, N_a0) * sys_mat.exc_bohr_radius_mat()
th_vec = linspace(0, pi, N_th)

file_id = '6_tzxtcdSWCrJR1QlfE9_A'

if file_id == '':
    ke_k_args = [(
        integ_pot,
        ke_k_pot,
        a0,
        sys_mat,
    ) for a0 in a0_vec]

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

    ke_vec = array(time_func(
        pool.starmap,
        time_func,
        ke_k_args,
    )).reshape((N_a0, ))

    starmap_args = [(
        calc_data,
        rho_cm,
        a0,
        th,
        sys_mat,
        n_a0,
        ke_vec,
    ) for (rho_cm, (n_a0, a0), th) in itertools.product(
        rho_cm_vec,
        enumerate(a0_vec),
        th_vec,
    )]

    data = array(time_func(
        pool.starmap,
        time_func,
        starmap_args,
    )).reshape((N_rho_cm, N_a0, N_th))

    pool.terminate()

    file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

    save_data(
        'extra/extcharge/pot_th_%s' % file_id,
        [data.flatten()],
    )
else:
    data = load_data(
        'extra/extcharge/pot_th_%s' % file_id,
        globals(),
    ).reshape((N_rho_cm, N_a0, N_th))

for n in range(N_a0):
    for n_rho in range(N_rho_cm):
        ax[n_rho].axhline(
            y=0,
            linestyle='-',
            linewidth=0.5,
            color='k',
        )

        ax[n_rho].plot(
            th_vec,
            data[n_rho, n, :],
            color=colors[n],
            linestyle='-',
            linewidth=1.5,
            label=r'%.1f' % (a0_vec[n] / sys_mat.exc_bohr_radius_mat()),
        )

for n_rho in range(N_rho_cm):
    ax[n_rho].set_xlim(0, pi)
    ax[n_rho].set_ylim(-3.5, 0.5)

    ax[n_rho].set_xticks([0, 0.25 * pi, 0.5 * pi, 0.75 * pi, pi])
    ax[n_rho].set_xticklabels([
        r'$0$',
        r'$\frac{\pi}{4}$',
        r'$\frac{\pi}{2}$',
        r'$\frac{3\pi}{4}$',
        r'$\pi$',
    ])

    ax[n_rho].set_xlabel(r'$\theta$')

    ax[n_rho].set_yticks([-3, -2, -1, 0])

    if n_rho > 0:
        ax[n_rho].set_yticks([])

ax[0].set_xticklabels(ax[0].get_xticklabels()[:-1] + [''])
ax[1].set_xticklabels([''] + ax[1].get_xticklabels()[1:-1] + [''])
ax[2].set_xticklabels([''] + ax[2].get_xticklabels()[1:])

ax[0].text(
    pi * 0.25,
    0.5 * 0.35,
    r'$\rho_{CM}~/~a_0$: $\frac{1}{10}$',
    color='k',
    fontsize=14,
)

ax[1].text(
    pi * 0.5,
    0.5 * 0.35,
    r'$1$',
    color='k',
    fontsize=14,
)

ax[2].text(
    pi * 0.5,
    0.5 * 0.35,
    r'$10$',
    color='k',
    fontsize=14,
)

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
    'pot_th_A1',
    transparent=True,
)

plt.show()
