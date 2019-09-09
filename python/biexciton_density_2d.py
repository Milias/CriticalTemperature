from common import *
import pyperclip

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

N_k = 1 << 10

eb_exc = -193e-3
eb_exc2 = -45e-3

fig_size = tuple(array([6.8, 5.3]) * 2)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

surf_area = 326.4  # nm^2
m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 1000  # K
sys = system_data(m_e, m_h, eps_r, T)

mu_e_lim = plasmon_exc_mu_val(eb_exc + 0.5 * eb_exc2, sys)
mu_h_lim = sys.get_mu_h(mu_e_lim)
mu_exc_lim = mu_e_lim + mu_h_lim
n_q_inf = sys.density_ideal(mu_e_lim)
n_exc_inf = sys.density_exc(mu_exc_lim, eb_exc)

print('mu_e_lim: %f, mu_exc_lim: %f' % (mu_e_lim, mu_exc_lim))
print('n_q_inf: %e, n_exc_inf: %e' % (n_q_inf, n_exc_inf))


def solve_u_f(mu_e, u, sys):
    return mu_e + sys.get_mu_h(
        mu_e) - eb_exc - 0.5 * eb_exc2 + 0.5 / sys.beta * log(1 + exp(u))


def solve_u(u, sys):
    mu_e_sol = root_scalar(
        solve_u_f,
        method='secant',
        x0=mu_e_lim,
        x1=2 * mu_e_lim,
        args=(u, sys),
        xtol=1e-10,
        options={'maxiter': 200},
    )

    return mu_e_sol.root


def eq_state(u, n_gamma, sys):
    mu_e = solve_u(u, sys)
    #print((u, mu_e))
    mu_h = sys.get_mu_h(mu_e)
    mu_exc = mu_e + mu_h

    n_q = sys.density_ideal(mu_e)
    n_exc = sys.density_exc(mu_exc, eb_exc)
    n_exc2 = sys.density_exc2_u(u)

    #print((n_q, n_exc, n_exc2))
    #print(n_q + n_exc + 2 * n_exc2 - n_gamma)

    return n_q + n_exc + 2 * n_exc2 - n_gamma


def solve_eq_state(n_gamma, sys):
    try:
        u_sol = root_scalar(
            eq_state,
            method='secant',
            x0=0.0,
            x1=1.0,
            xtol=1e-10,
            args=(n_gamma, sys),
            options={'maxiter': 200},
        )

        if not u_sol.converged:
            print('Did not converge for (%e,%e)' % (n_gamma, sys.T))

        return (solve_u(u_sol.root, sys), u_sol.root)
    except Exception as e:
        print('Failed for (%e,%e): %s' % (n_gamma, sys.T, e))
        return (float('nan'), float('nan'))


N_n, N_T = 128, 128

n_gamma_vec = logspace(-2.5, 1, N_n) / sys.a0**2
T_vec = logspace(1.8, 3.5, N_T)
result = zeros((n_gamma_vec.size, T_vec.size, 5))
plot_image = zeros((N_T, N_n, 3))

n_q_color = array([0.8, 0.1, 0.1])
n_exc_color = array([0.1, 0.8, 0.1])
n_exc2_color = array([0.1, 0.1, 0.8])

for i, j in itertools.product(range(N_n), range(N_T)):
    n_gamma, sys = n_gamma_vec[i], system_data(m_e, m_h, eps_r, T_vec[j])
    mu_e, u = solve_eq_state(n_gamma, sys)

    if isnan(mu_e):
        mu_e = mu_e_lim
        mu_h = mu_h_lim

        n_q = 0
        n_exc = 0
        n_exc2 = 0
    else:
        mu_h = sys.get_mu_h(mu_e)
        mu_exc = mu_e + mu_h

        n_q = sys.density_ideal(mu_e)
        n_exc = sys.density_exc(mu_exc, eb_exc)
        n_exc2 = sys.density_exc2_u(u)

    #print((n_q, n_exc, n_exc2))

    result[i, j, 0] = mu_e
    result[i, j, 1] = mu_h
    result[i, j, 2] = n_q * sys.a0**2
    result[i, j, 3] = n_exc * sys.a0**2
    result[i, j, 4] = n_exc2 * sys.a0**2

    plot_image[j, i] = n_q / n_gamma * n_q_color + \
        n_exc / n_gamma * n_exc_color + \
        2 * n_exc2 / n_gamma * n_exc2_color

n_gamma_vec = n_gamma_vec * sys.a0**2

X, Y = meshgrid(n_gamma_vec, T_vec)

color_tuples = array([
    plot_image[:-1, :-1, 0].flatten(),
    plot_image[:-1, :-1, 1].flatten(),
    plot_image[:-1, :-1, 2].flatten(),
]).transpose()

ax[0].set_xscale('log')
ax[0].set_yscale('log')

im = ax[0].pcolormesh(
    X,
    Y,
    zeros_like(X),
    color=color_tuples,
    snap=True,
)
im.set_array(None)

ax[0].plot(
    [6.27 / surf_area * sys.a0**2, 52.85 / surf_area * sys.a0**2],
    [294, 294],
    '-',
    color='m',
)

ax[0].plot(
    [6.27 / surf_area * sys.a0**2, 52.85 / surf_area * sys.a0**2],
    [294, 294],
    'o',
    markeredgecolor='m',
    markerfacecolor='#FFFFFF',
)

ax[0].set_xlabel(r'$n_\gamma a_0^2$')
ax[0].set_ylabel(r'$T$ (K)')

plt.tight_layout()
plt.savefig('plots/papers/biexciton1/%s.pdf' % 'biexciton_diagram')
plt.show()
