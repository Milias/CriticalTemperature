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


N_rho_cm = 128
N_a0 = 128
N_th = 1

size_d = 1  # nm
eps_sol = 1
eps_mat = 6 * eps_sol
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_mat = system_data(m_e, m_h, eps_sol, T, size_d, eps_mat, ext_dist_l)

file_id = 'ejmCvqDNRoSl5eyqod0XJA'


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


rho_cm_vec = logspace(log10(0.05), log10(20), N_rho_cm)
a0_vec = logspace(log10(0.2), log10(10), N_a0)
th_vec = array([pi])

if file_id == '':
    ke_k_args = [(
        integ_pot,
        ke_k_pot,
        a0 * sys_mat.exc_bohr_radius_mat(),
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
        rho_cm * sys_mat.exc_bohr_radius_mat(),
        a0 * sys_mat.exc_bohr_radius_mat(),
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
        'extra/extcharge/pot_2d_%s' % file_id,
        [data.flatten()],
    )
else:
    data = load_data(
        'extra/extcharge/pot_2d_%s' % file_id,
        globals(),
    ).reshape((N_rho_cm, N_a0, N_th))

X, Y = meshgrid(rho_cm_vec, a0_vec)

hsv_colors = array([[0.3, 0.8, 0.8]])
colors = array([matplotlib.colors.hsv_to_rgb(c) for c in hsv_colors])
cm = ListedColormap([x * colors[0] for x in linspace(1, 0, 256)])

im = ax[0].pcolormesh(
    X,
    Y,
    data[:, :, 0].T,
    cmap=cm,
    snap=True,
    antialiased=True,
    rasterized=True,
)

ax[0].axhline(
    y=1,
    color='w',
    linestyle='--',
    dashes=(3., 5.),
    dash_capstyle='round',
    linewidth=0.6,
)

ax[0].axhline(
    y=3,
    color='w',
    linestyle='--',
    dashes=(3., 5.),
    dash_capstyle='round',
    linewidth=0.6,
)

n_ticks = 6

cb = fig.colorbar(
    ScalarMappable(cmap=cm),
    ax=ax[0],
    boundaries=linspace(amin(data), amax(data), 256),
    ticks=linspace(amin(data), amax(data), n_ticks),
    format=r'$%.1f$',
    fraction=0.05,
    pad=0.01,
)

ax[0].set_xlim(rho_cm_vec[0], rho_cm_vec[-1])
ax[0].set_ylim(a0_vec[0], a0_vec[-1])

ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[0].set_xlabel(r'$\rho_{CM}~/~a_0$')
ax[0].set_ylabel(r'$|\vec{\rho}_e-\vec{\rho}_h|~/~a_0$')

plt.tight_layout()

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'pot_2d_A1',
    transparent=True,
)

plt.show()
