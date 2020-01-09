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
    if abs(sys.eps_mat - sys.eps_r) < 1e-8:
        pot = 1
    else:
        pot = 1 / tanh(u * sys.size_d * 0.5 / x +
                       eta_func(sys.eps_mat, sys.eps_r))

    return -special.j0(u) * pot / x / sys.eps_mat * sys.c_aEM * sys.c_hbarc


def hn_r_pot_norm(x, sys):
    return -pi * (special.struve(0, 2 * x / size_d_eff(sys)) -
                  special.y0(2 * x / size_d_eff(sys)))


N_x, N_eps = 128, 1

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.22, 0.41, 294  # K

eps_vec = eps_sol / array([0.1])

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_vec = [system_data(m_e, m_h, eps_sol, T, size_d, eps) for eps in eps_vec]


def integ_pot(pot_func, x, sys):
    return sum([
        quad(
            pot_func,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, sys),
        )[0] for n in arange(1e4)
    ])


#x_vec = linspace(5e-3, 4 * eps / eps_sol * size_d, 256)
x_vec = logspace(log10(2e-2), log10(40), N_x)

integ_ke_args = [(
    integ_pot,
    ke_k_pot,
    x,
    sys,
) for x, sys in itertools.product(x_vec, sys_vec)]

pool = multiprocessing.Pool(multiprocessing.cpu_count())

ke_vec = array(time_func(
    pool.starmap,
    time_func,
    integ_ke_args,
)).reshape((N_x, N_eps))

pool.terminate()

hn_vec = array([hn_r_pot_norm(x_vec, sys) for sys in sys_vec]).reshape(
    (N_eps, N_x)).T

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, eps_vec.size)
]

ax[0].axvline(
    x=1,
    linestyle='-',
    color='g',
    linewidth=0.9,
)

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    ax[0].axvline(
        x=size_d / size_d_eff(sys),
        linestyle='-',
        color='b',
        linewidth=0.7,
    )

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

    ax[0].semilogx(
        x_vec,
        ke_vec[:, n_eps] / energy_norm,
        color=c,
        linestyle='-',
        linewidth=2.0,
        label=r'$V^{RK}(r)$',
        zorder=10,
    )

    ax[0].semilogx(
        x_vec[x_vec < 0.09],
        -size_d / x_vec[x_vec < 0.09],
        color='b',
        linestyle='-',
        linewidth=0.8,
        label=r'$V^{C}(r)$',
    )

    ax[0].semilogx(
        x_vec[x_vec > 0.11],
        hn_vec[x_vec > 0.11, n_eps],
        color='m',
        linestyle='--',
        dashes=(0.8, 4.),
        dash_capstyle='round',
        linewidth=1.0,
        label=r'$V^{\mathcal{H}N}(r)$',
    )

    ax[0].semilogx(
        x_vec[x_vec > 1.1],
        -size_d_eff(sys) / x_vec[x_vec > 1.1],
        color='g',
        linestyle='--',
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=1.0,
        label=r'$V_{sol}^{C}(r)$',
    )

ax[0].text(
    0.032,
    -15,
    r'$-\frac{d}{r}$',
    color='b',
    fontsize=28,
)

ax[0].text(
    0.14,
    -6,
    r'$V^{\mathcal{H}N}$',
    color='m',
    fontsize=24,
)

ax[0].text(
    2,
    -8,
    r'$-\frac{d^*}{r}$',
    color='g',
    fontsize=28,
)

ax[0].set_yticks([0])
ax[0].yaxis.set_label_coords(-0.02, 0.5)

ax[0].set_xticks([0.1, 1])
ax[0].set_xticks([], minor=True)
ax[0].set_xticklabels([r'$d$', r'$d^*$'])
ax[0].xaxis.tick_top()

ax[0].set_ylabel(r'$V(r)$')
ax[0].set_xlabel(r'$r$')

ax[0].set_ylim(-25, 0)
ax[0].set_xlim(x_vec[0], x_vec[-1])

ax[0].legend(loc=0)

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'pot_scheme_v1')

plt.show()
