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


N_x, N_eps = 32, 3

size_d = 1  # nm
eps_sol = 1
m_e, m_h, T = 0.27, 0.45, 294  # K

eps_vec = eps_sol / array([1.0, 0.5, 0.2])

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
        )[0] for n in arange(0.5e4)
    ])


#x_vec = linspace(5e-3, 4 * eps / eps_sol * size_d, 256)
x_vec = logspace(log10(1e-2), log10(10), N_x)

integ_ke_args = [(
    integ_pot,
    ke_k_pot,
    x * size_d_eff(sys) / sys.size_d,
    sys,
) for x, sys in itertools.product(x_vec, sys_vec)]

x_vec2 = logspace(log10(1e-2), log10(10), N_x)
integ_ke_args2 = [(
    integ_pot,
    ke_k_pot,
    x,
    sys,
) for x, sys in itertools.product(x_vec2, sys_vec)]

pool = multiprocessing.Pool(multiprocessing.cpu_count())

y_vec = array(time_func(
    pool.starmap,
    time_func,
    integ_ke_args,
)).reshape((N_x, N_eps))

y_vec2 = array(time_func(
    pool.starmap,
    time_func,
    integ_ke_args2,
)).reshape((N_x, N_eps))

pool.terminate()

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, eps_vec.size)
]

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    ax[0].axvline(
        x=size_d / size_d_eff(sys),
        linestyle='--',
        color=c,
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.8,
    )
    ax[1].axvline(
        x=size_d_eff(sys) / size_d,
        linestyle='--',
        color=c,
        dashes=(3., 5.),
        dash_capstyle='round',
        linewidth=0.8,
    )

for (n_eps, sys), c in zip(enumerate(sys_vec), colors):
    energy_norm = sys_sol.c_aEM * sys_sol.c_hbarc / sys.eps_mat / sys.size_d

    ax[0].semilogx(
        x_vec,
        y_vec[:, n_eps] / energy_norm,
        '-',
        color=c,
        label=r'$d / d^*$: $%.1f$' % (size_d / size_d_eff(sys)),
    )
    ax[1].semilogx(
        x_vec2,
        y_vec2[:, n_eps] / energy_norm,
        '-',
        color=c,
        label=r'$d^* / d$: $%d$' % (size_d_eff(sys) / size_d),
    )

ax[0].axhline(y=0, color='k', linewidth=0.7)

ax[0].set_yticks([0])
ax[1].set_yticks([])
ax[0].yaxis.set_label_coords(-0.02, 0.5)

ax[0].set_ylabel(r'$V^{RK}(r)$')
ax[0].set_xlabel(r'$r / d^*$')
ax[1].set_xlabel(r'$r / d$')

ax[0].legend(title=r'$d / d^* = \epsilon_{sol} / \epsilon$')
ax[1].legend(title=r'$d^* / d = \epsilon / \epsilon_{sol}$')

ax[0].set_ylim(-20, 0)
ax[1].set_ylim(-20, 0)
ax[0].set_xlim(x_vec[0], x_vec[-1] * 0.99)
ax[1].set_xlim(x_vec2[0] * 1.01, x_vec2[-1])

plt.tight_layout()

fig.subplots_adjust(wspace=0)

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'pot_comparison_B4')

plt.show()
