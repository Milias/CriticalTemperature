from common import *
import pyperclip

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

from matplotlib.legend_handler import HandlerBase


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height,
                       fontsize, trans):
        l1 = plt.Line2D([x0, y0 + width], [0.7 * height, 0.7 * height],
                        linestyle=orig_handle[1],
                        color=orig_handle[0])
        l2 = plt.Line2D([x0, y0 + width], [0.3 * height, 0.3 * height],
                        color=orig_handle[0])
        return [l1, l2]


N_k = 1 << 10

eb_exc = -193e-3
eb_exc2 = -45e-3

fig_size = tuple(array([6.8, 5.3]) * 2)

n_x, n_y = 1, 2
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

surf_area = 326.4  # nm^2
m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 1000  # K
sys = system_data(m_e, m_h, eps_r, T)

mu_e_lim = plasmon_exc_mu_val(eb_exc + 0.5 * eb_exc2, sys)
#mu_e_lim = plasmon_exc_mu_val(eb_exc, sys)

mu_exc_lim = mu_e_lim + sys.get_mu_h(mu_e_lim)

print('mu_e_lim: %f, mu_exc_lim: %f' % (mu_e_lim, mu_exc_lim))


def eq_state(u, n_gamma, sys):
    mu_e = mu_e_lim * (1.0 + exp(u[0]))

    mu_h = sys.get_mu_h(mu_e)
    mu_exc = mu_e + mu_h

    n_q = sys.density_ideal(mu_e)
    n_exc = sys.density_exc(mu_exc, eb_exc)
    n_exc2 = sys.density_exc2(mu_exc, eb_exc, eb_exc2)

    return n_q + n_exc + 2 * n_exc2 - n_gamma


def solve_eq_state(n_gamma, sys):
    u_sol = root(
        eq_state,
        [1.4],
        method='broyden1',
        tol=1e-10,
        args=(n_gamma, sys),
        options={'maxiter': 1000},
    )

    return mu_e_lim * (1.0 + exp(u_sol.x[0]))


n_gamma_vec = logspace(-2.8, 0, 32)
result = zeros((n_gamma_vec.size, 5))

for i, n_gamma in enumerate(n_gamma_vec):
    mu_e = solve_eq_state(n_gamma, sys)

    mu_h = sys.get_mu_h(mu_e)
    mu_exc = mu_e + mu_h

    n_q = sys.density_ideal(mu_e)
    n_exc = sys.density_exc(mu_exc, eb_exc)
    n_exc2 = sys.density_exc2(mu_exc, eb_exc, eb_exc2)

    result[i, 0] = mu_e
    result[i, 1] = mu_h
    result[i, 2] = n_q * sys.a0**2
    result[i, 3] = n_exc * sys.a0**2
    result[i, 4] = n_exc2 * sys.a0**2

    print(abs(n_q + n_exc + 2 * n_exc2 - n_gamma))

plot_type = 'loglog'

n_gamma_vec = n_gamma_vec * sys.a0**2

getattr(ax[0], plot_type)(
    n_gamma_vec,
    result[:, 2],
    '--',
    color='r',
    label=r'$n_q$, T=%.0f' % T,
)

getattr(ax[0], plot_type)(
    n_gamma_vec,
    result[:, 3],
    '-',
    color='r',
    label=r'$n_{exc}$',
)

getattr(ax[0], plot_type)(
    n_gamma_vec,
    result[:, 4],
    '-.',
    color='r',
    label=r'$n_{exc_2}$',
)

getattr(ax[0], plot_type)(
    n_gamma_vec,
    result[:,3] + result[:, 4],
    '-',
    color='g',
    label=r'$n_{exc} + n_{exc_2}$',
)

ax[0].axvline(x=6.27 / surf_area * sys.a0**2, color='m', linestyle=':')
ax[0].axvline(x=52.85 / surf_area * sys.a0**2, color='m', linestyle=':')

ax[0].legend(loc=0)

plot_type = 'semilogx'

getattr(ax[1], plot_type)(
    n_gamma_vec,
    result[:, 0],
    '-',
    color='g',
    label=r'$\mu_e$'
)

getattr(ax[1], plot_type)(
    n_gamma_vec,
    result[:, 0] + result[:, 1],
    '-',
    color='b',
    label=r'$\mu_{exc}$'
)

ax[1].axhline(y=mu_e_lim, color='g', linestyle='--')
ax[1].axhline(y=mu_exc_lim, color='b', linestyle='--')
ax[1].axvline(x=6.27 / surf_area * sys.a0**2, color='m', linestyle=':')
ax[1].axvline(x=52.85 / surf_area * sys.a0**2, color='m', linestyle=':')

ax[1].legend(loc=0)

plt.tight_layout()
plt.show()
