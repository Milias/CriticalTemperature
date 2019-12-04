from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]))

n_x, n_y = 2, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]


def ke_k_pot(u, x, size_d, eta):
    return special.j0(u) / tanh(u * size_d * 0.5 / x + eta)


size_d = 1.0
eps_sol = 1.0
#eps = (1 + 1e-1) * eps_sol
eps = 3.0
eta = 0.5 * log((eps + eps_sol) / (eps - eps_sol))


def integ_ke_k_pot(x):
    return -sum([
        quad(
            ke_k_pot,
            4 * n * pi,
            4 * (n + 1) * pi,
            limit=500,
            args=(x, size_d, eta),
        )[0] for n in arange(1e4)
    ]) / x / eps


#x_vec = linspace(5e-3, 4 * eps / eps_sol * size_d, 256)
x_vec = logspace(log10(3e-3), log10(10 * eps / eps_sol * size_d), 256)

pool = multiprocessing.Pool(32)
y_vec = array(pool.map(integ_ke_k_pot, x_vec))
pool.terminate()

cou_short_vec = -1 / x_vec / eps
cou_long_vec = -1 / x_vec / tanh(eta) / eps

ax[0].axvline(x=size_d, color='m', linewidth=0.7)
ax[0].axhline(y=0, color='k', linewidth=0.7)

ax[0].semilogx(x_vec, y_vec, 'r-', label='RK potential')
ax[0].semilogx(x_vec[x_vec < eps / eps_sol * size_d],
               cou_short_vec[x_vec < eps / eps_sol * size_d],
               'g--',
               label='Coulomb plate')
ax[0].semilogx(x_vec[x_vec > size_d / (eps / eps_sol)],
               cou_long_vec[x_vec > size_d / (eps / eps_sol)],
               'b--',
               label='Coulomb solution')

ax[1].semilogx(
    x_vec,
    -1 / (y_vec * x_vec),
    'r-',
)

ax[0].set_yticks([0])
ax[0].set_yticklabels(['$0$'])
ax[0].yaxis.set_label_coords(-0.02, 0.5)

ax[1].set_yticks([eps_sol,eps])
ax[1].set_yticklabels([r'$\epsilon_{sol}$', r'$\epsilon$'])
ax[1].yaxis.set_label_coords(-0.02, 0.5)

ax[0].set_ylabel('$V(r)$')
ax[1].set_ylabel(r'$\epsilon(r)$')
ax[1].set_xlabel(r'$r d^{-1}$')

ax[0].legend()

ax[1].axhline(y=eps, color='g', linewidth=0.7)
ax[1].axhline(y=eps_sol, color='b', linewidth=0.7)
ax[1].axvline(x=size_d, color='m', linewidth=0.7)

ax[0].set_ylim(-12, 0)
ax[0].set_xlim(x_vec[0], x_vec[-1])
ax[1].set_xlim(x_vec[0], x_vec[-1])

plt.tight_layout()

plt.setp(ax[0].get_xticklabels(), visible=False)
fig.subplots_adjust(hspace=0)

plt.savefig('/storage/Reference/Work/University/PhD/Keldysh/%s.pdf' %
            'pot_comparison')

plt.show()

