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
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]


def size_d_eff(sys):
    return sys.size_d * sys.eps_mat / sys.eps_r


def eta_func(eps_mat, eps_sol):
    return 0.5 * log((eps_mat + eps_sol) / (eps_mat - eps_sol))


def ke_k_pot(u, x, sys):
    pot = (sys.eps_mat + sys.eps_r * tanh(u * sys.size_d / x * 0.5)) / (
        sys.eps_r + sys.eps_mat * tanh(u * sys.size_d / x * 0.5))

    return -special.j0(u) * pot / (x * sys.eps_mat) * sys.c_aEM * sys.c_hbarc


def ext_k_pot(u, x, sys):
    pot = 2 * exp(-u * (sys.ext_dist_l + 0.5 * sys.size_d) /
                  x) / (sys.eps_mat + sys.eps_r +
                        (sys.eps_r - sys.eps_mat) * exp(-u * sys.size_d / x))

    return -special.j0(u) * pot / x * sys.c_aEM * sys.c_hbarc


def hn_r_pot_norm(x, sys):
    return -pi * (special.struve(0, 2 * x / size_d_eff(sys)) -
                  special.y0(2 * x / size_d_eff(sys)))


N_x, N_eps = 64, 3

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

eps_vec = eps_sol / array([2, 1, 0.5])

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_vec = [
    system_data(m_e, m_h, eps_sol, T, size_d, eps, ext_dist_l)
    for eps in eps_vec
]

a0 = 0.01

def integ_pot(pot_func, x, sys):
    return sum([
        quad(
            pot_func,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, sys),
        )[0] for n in arange(0.5e4)
    ])


x_vec = logspace(log10(1e-3), log10(4), N_x)

integ_ext_e_args = [(
    integ_pot,
    ext_k_pot,
    x,
    sys,
) for x, sys in itertools.product(x_vec, sys_vec)]

integ_ext_h_args = [(
    integ_pot,
    ext_k_pot,
    x + a0,
    sys,
) for x, sys in itertools.product(x_vec, sys_vec)]

pool = multiprocessing.Pool(multiprocessing.cpu_count())

ext_e_vec = array(time_func(
    pool.starmap,
    time_func,
    integ_ext_e_args,
)).reshape((N_x, N_eps))

ext_h_vec = array(time_func(
    pool.starmap,
    time_func,
    integ_ext_h_args,
)).reshape((N_x, N_eps))

pool.terminate()

ke_vec = array(
    [integ_pot(ke_k_pot, a0, sys) for sys in sys_vec])

hn_vec = array([hn_r_pot_norm(x_vec, sys) for sys in sys_vec]).reshape(
    (N_eps, N_x)).T

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, eps_vec.size)
]

ax[0].axvline(
    x=a0,
    linestyle='-',
    color='m',
    linewidth=0.9,
)

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

    ax[0].plot(
        x_vec,
        (ke_vec[n_eps] + ext_e_vec[:, n_eps] - ext_h_vec[:, n_eps]) /
        energy_norm,
        color=c,
        linestyle='-',
        linewidth=2.0,
        zorder=10,
    )

ax[0].set_ylabel(r'$V(r)$')
ax[0].set_xlabel(r'$r$')

#ax[0].set_ylim(-25, 0)
ax[0].set_xlim(0, x_vec[-1])

#ax[0].legend(loc = 0)

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
            'r_pot_A1')

plt.show()
