from common import *

N_k = 1 << 8
eb_cou = 0.193

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 294  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
sys = system_data(m_e, m_h, eps_r, T)

u_vec = linspace(-3, 1, 8)
mu_e_vec = linspace(-0.2, -8.083269038076584811e-02, 128)
mu_h_vec = array([sys.get_mu_h(mu_e) for mu_e in mu_e_vec])

mu_e_data, eb_data = loadtxt('extra/eb_vec_int.csv', delimiter=',').T

if mu_e_data.size != mu_e_vec.size:
    eb_vec = array(time_func(plasmon_det_zero_ht_v, N_k, mu_e_vec, sys))

    export_eb_data = zeros((eb_vec.size, 2))
    export_eb_data[:, 0] = mu_e_vec
    export_eb_data[:, 1] = eb_vec

    savetxt('extra/eb_vec_int.csv', export_eb_data, delimiter=',')
else:
    eb_vec = eb_data

mu_exc = eb_vec - mu_e_vec - mu_h_vec

colors = [
    matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb([h, 0.8, 0.8]))
    for h in linspace(0, 0.7, u_vec.size)
]

n_x, n_y = 2, 1
fig = plt.figure(figsize=(5.8, 8.3), dpi=150)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

plot_func = {'log': ('loglog', 'semilogx'), 'linear': ('plot', 'plot')}

plot_type = 'linear'

getattr(ax[1], plot_func[plot_type][1])(
    mu_e_vec,
    mu_exc,
    '-',
    color='k',
    #label='u: $%f$' % u,
)

ax[1].axhline(y=0, color='k')

for c, (i, u) in zip(colors, enumerate(u_vec)):
    n_exc = sys.density_exc_exp(u)

    log_arg = 4 * sys.c_hbarc**2 / abs(mu_e_vec + mu_h_vec - eb_vec) / (
        sys.m_e + sys.m_h) * n_exc #/ sys.a0**2

    V0_vec = 4 * pi * sys.c_hbarc**2 / (sys.m_e + sys.m_h) / log(log_arg)

    energy_density_vec = V0_vec * n_exc
    v0n = energy_density_vec - logExp(u) / sys.beta

    getattr(ax[0], plot_func[plot_type][0])(
        mu_e_vec,
        log_arg,
        '-',
        color=c,
    )

    ax[0].axhline(y=1, color='g')
    #ax[0].legend(loc=0)

    getattr(ax[1], plot_func[plot_type][1])(
        mu_e_vec,
        v0n,
        '-',
        color=c,
        label='$n_{exc} = %.2f$ nm$^{-2}$' % n_exc,
    )

    ax[1].axhline(y=logExp(u) / sys.beta, color=c, linestyle='--')
    ax[1].set_ylim(amin(mu_exc), amax(mu_exc))
    ax[1].legend(loc=0)

fig.tight_layout()
plt.show()
