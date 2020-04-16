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


def ke_k_pot(u, x, sys):
    pot = (sys.eps_mat + sys.eps_r * tanh(u * sys.size_d / x * 0.5)) / (
        sys.eps_r + sys.eps_mat * tanh(u * sys.size_d / x * 0.5))

    return -special.j0(u) * pot / (x * sys.eps_mat) * sys.c_aEM * sys.c_hbarc


def ext_k_pot(u, x, sys):
    pot = 2 * exp(-u * (sys.ext_dist_l + 0.5 * sys.size_d) /
                  x) / (sys.eps_mat + sys.eps_r +
                        (sys.eps_r - sys.eps_mat) * exp(-u * sys.size_d / x))

    return -special.j0(u) * pot / x * sys.c_aEM * sys.c_hbarc


N_eps = 5
N_rho_cm = 1 << 9
N_a0 = N_eps
N_th = 1

size_d = 1  # nm
eps_sol = 2
eps_vec = logspace(0, log10(10), N_eps) * eps_sol
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_vec = [
    system_data(m_e, m_h, eps_sol, T, size_d, eps, ext_dist_l)
    for eps in eps_vec
]

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, N_eps)
]

file_id = 'NjqwIj6gQeyATcDfRcLdRA'


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


rho_cm_vec = linspace(0, 4, N_rho_cm)
a0_vec = array([sys.exc_bohr_radius_mat() for sys in sys_vec])
th_vec = array([pi])

if file_id == '':
    ke_k_args = [(
        integ_pot,
        ke_k_pot,
        a0,
        sys,
    ) for a0, sys in zip(a0_vec, sys_vec)]

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
        sys,
        n_a0,
        ke_vec,
    ) for (rho_cm, ((n_a0, a0), sys), th) in itertools.product(
        rho_cm_vec,
        zip(enumerate(a0_vec), sys_vec),
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
        'extra/extcharge/pot_rcm_%s' % file_id,
        [data.flatten()],
    )
else:
    data = load_data(
        'extra/extcharge/pot_rcm_%s' % file_id,
        globals(),
    ).reshape((N_rho_cm, N_a0, N_th))

for n in range(N_eps):
    ax[0].axvline(
        x=a0_vec[n] / (1 + sys_vec[n].m_eh),
        color=colors[n],
        linestyle='-',
        linewidth=0.6,
    )

for n in range(N_eps):
    ax[0].plot(
        rho_cm_vec,
        data[:, n, 0],
        color=colors[n],
        linestyle='-',
        linewidth=2,
        label='%.1f' % (eps_vec[n] / eps_sol),
    )

ax[0].set_xlim(0, rho_cm_vec[-1])
ax[0].set_ylim(-3, 0)

ax[0].set_yticks([-3, -2, -1, 0])
ax[0].set_xticks([0, 1, 2, 3, 4])

ax[0].set_ylabel(r'$V(\rho_{CM})$ (eV)')
ax[0].set_xlabel(r'$\rho_{CM}$ (nm)')
ax[0].legend(loc='lower right', title=r'$\epsilon~/~\epsilon_{sol}$')

plt.tight_layout()

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'pot_rcm_A1',
    transparent=True,
)

plt.show()
