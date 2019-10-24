from common import *
import pyperclip

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

N_k = 1 << 10

be_exc = -193e-3
be_biexc = -45e-3

surf_area = 326.4  # nm^2
m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 1000  # K
sys = system_data(m_e, m_h, eps_r, T)

mu_e_lim = sys.exc_mu_val(be_exc + 0.5 * be_biexc)
mu_h_lim = sys.get_mu_h(mu_e_lim)
mu_exc_lim = mu_e_lim + mu_h_lim
n_q_inf = sys.density_ideal(mu_e_lim)
n_exc_inf = sys.density_exc(mu_exc_lim, be_exc)

fig_size = tuple(array([6.8, 5.3]) * 2)

n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]


def solve_eq_state(n_gamma, sys):
    return result_s(biexciton_eqst_c(n_gamma, be_exc, be_biexc, 0.0,
                                     sys)).value


def solve_eqst_data(T_vec, N_vec):
    T_arr, N_arr = meshgrid(T_vec, N_vec)
    N_T_arr = zeros((N_arr.size, 2))
    N_T_arr[:, 0] = N_arr.ravel()
    N_T_arr[:, 1] = T_arr.ravel()

    return array(
        biexciton_eqst_c_vec(
            N_T_arr.ravel(),
            be_exc,
            be_biexc,
            0.0,
            sys,
        )).reshape((N_arr.size, 2))


def solve_eqst_cT(cT, n0, n1, c):
    cT_sys = system_data(m_e, m_h, eps_r, cT)
    cT_n_vec = logspace(n0, n1, 1 << 7) / sys.a0**2
    mu_e_u_cT_data = solve_eqst_data(
        array([cT]),
        cT_n_vec,
    )
    cT_mu_exc = array(
        [mu_e + cT_sys.get_mu_h(mu_e) for mu_e in mu_e_u_cT_data[:, 0]])

    ax[0].plot(cT_n_vec * cT_sys.a0**2, [cT] * cT_n_vec.size, '-', color=c)
    ax[1].semilogx(cT_n_vec * cT_sys.a0**2,
               cT_sys.beta * (cT_mu_exc - be_exc),
               '--',
               color=c,
               label=r'$\mu_{exc} - E_{B,X}$')
    ax[1].semilogx(cT_n_vec * cT_sys.a0**2,
               cT_sys.beta * (2 * (cT_mu_exc - be_exc) - be_biexc),
               ':',
               color=c,
               label=r'$2 \mu_{exc} - 2 E_{B,X} - E_{B,X_2}$')


N_n, N_T = 1 << 9, 1 << 9

n_gamma_vec = logspace(-1.5, 2.7, N_n) / sys.a0**2
T_vec = logspace(1.8, 3.5, N_T)
result = zeros((n_gamma_vec.size, T_vec.size, 5))
plot_image = zeros((N_T, N_n, 3))

n_q_color = array([0.8, 0.1, 0.1])
n_exc_color = array([0.1, 0.8, 0.1])
n_exc2_color = array([0.1, 0.1, 0.8])
n_exc_deg_color = array([0.8, 0.8, 0.1])
n_exc2_deg_color = array([0.8, 0.1, 0.8])

mu_e_u_data = solve_eqst_data(T_vec, n_gamma_vec)

[solve_eqst_cT(T, 0.2, 2.0, c) for T, c in [(1e3, 'r'), (2e3, 'b'), (3e3, 'g')]]

ax[1].axhline(y=0, color='k', linewidth=0.2)
ax[1].legend()

for enum_n, (i, j) in enumerate(itertools.product(range(N_n), range(N_T))):
    n_gamma, sys = n_gamma_vec[i], system_data(m_e, m_h, eps_r, T_vec[j])
    mu_e, u = mu_e_u_data[enum_n]

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
        n_exc = sys.density_exc(mu_exc, be_exc)
        n_exc2 = sys.density_exc2_u(u)

    result[i, j, 0] = mu_e
    result[i, j, 1] = mu_h
    result[i, j, 2] = n_q * sys.lambda_th**2
    result[i, j, 3] = n_exc * sys.lambda_th**2
    result[i, j, 4] = n_exc2 * sys.lambda_th_biexc**2

    plot_image[j, i] = n_q / n_gamma * n_q_color
    plot_image[j, i] += n_exc / n_gamma * (n_exc_color if result[i, j, 3] < 4
                                           else n_exc_deg_color)
    plot_image[j, i] += n_exc2 / n_gamma * (n_exc2_color if result[i, j, 4] < 1
                                            else n_exc2_deg_color)

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

print('mu_e_lim: %f, mu_exc_lim: %f' % (mu_e_lim, mu_exc_lim))
print('n_q_inf: %e, n_exc_inf: %e' % (n_q_inf, n_exc_inf))

plt.tight_layout()
#"""
plt.savefig('plots/papers/biexciton1/%s.png' % 'biexciton_diagram_mu', dpi = 600)
plt.savefig('plots/papers/biexciton1/%s.pdf' % 'biexciton_diagram_mu')
#"""
plt.show()
