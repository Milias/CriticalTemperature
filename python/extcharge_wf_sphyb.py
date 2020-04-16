from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([1.5 * 6.8, 2 * 5.3]))

n_x, n_y = 3, 3
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


N_rho_cm = 1
N_a0 = 128
N_th = 128

size_d = 1  # nm
eps_sol = 1
eps_mat = 6 * eps_sol
m_e, m_h, T = 0.27, 0.45, 294  # K

ext_dist_l = 0

sys_sol = system_data(m_e, m_h, eps_sol, T, size_d, eps_sol)
sys_mat = system_data(m_e, m_h, eps_sol, T, size_d, eps_mat, ext_dist_l)

file_id = ''


def psi10(r, th, sys):
    a0 = sys.exc_bohr_radius_mat()
    return 0.5 / a0 * exp(-r / a0) / sqrt(2 * pi)


def psi20(r, th, sys):
    a0 = sys.exc_bohr_radius_mat()
    return 2 / 3 / sqrt(3) / a0 * (1 - 2 * r / 3 / a0) * exp(
        -r / a0 / 3) / sqrt(2 * pi)


def psi21(r, th, sys):
    a0 = sys.exc_bohr_radius_mat()
    return 2 * sqrt(2) / 9 / sqrt(3) / a0**2 * r * exp(-r / a0 / 3) / sqrt(
        2 * pi) * exp(1j * th)


def psi2_1(r, th, sys):
    return conj(psi21(r, th, sys))


rho_cm_vec = array([1])
a0_vec = linspace(0, 16, N_a0) * sys_mat.exc_bohr_radius_mat()
th_vec = linspace(0, 2 * pi, N_th)

R, TH = meshgrid(a0_vec, th_vec)
X, Y = R * cos(TH), R * sin(TH)

psi_orb = [
    #psi21(R, TH, sys_mat),
    (psi21(R, TH, sys_mat) + psi2_1(R, TH, sys_mat)) * sqrt(0.5),
    psi10(R, TH, sys_mat),
]

mix_vec = linspace(-1, 1, n_x * n_y)

psi_vec = [(psi_orb[0] * x + psi_orb[1] * (1 - x)) if x >= 0 else
           (psi_orb[0] * x + psi_orb[1] * (1 + x)) for x in mix_vec]

mod_vec = [real(abs(psi)) for psi in psi_vec]
"""
angle_vec = [real(angle(psi)) for psi in psi_vec]

data_vec = [
    array([
        mod.ravel(),
        ang.ravel(),
        0.8 * ones_like(mod.ravel()),
    ]).reshape((3, N_a0 * N_th)).T for (mod, ang) in zip(mod_vec, angle_vec)
]

print(data_vec[0])
print(data_vec[0].shape)
"""

data_vec = mod_vec

hsv_colors = array([[0.3, 0.8, 0.8]])
colors = array([matplotlib.colors.hsv_to_rgb(c) for c in hsv_colors])
cm = ListedColormap([x * colors[0] for x in linspace(0, 1, 256)])

for n, d in enumerate(data_vec):
    im = ax[n].pcolormesh(
        X,
        Y,
        d,
        cmap=cm,
        snap=True,
        antialiased=True,
        rasterized=True,
    )

    x_max = a0_vec[-1] * cos(pi * 0.25)
    ax[n].set_xlim(-x_max, x_max)
    ax[n].set_ylim(-x_max, x_max)
    ax[n].set_xticks([])
    ax[n].set_yticks([])
    ax[n].set_aspect('equal')

    ax[n].text(
        0,
        0.85 * x_max,
        r'$%.2f$' % mix_vec[n],
        color='w',
        fontsize=24,
        horizontalalignment='center',
        verticalalignment='center',
    )

plt.tight_layout()

fig.subplots_adjust(wspace=0.01, hspace=0.01)

plt.savefig(
    '/storage/Reference/Work/University/PhD/ExternalCharge/%s.pdf' %
    'pot_wf_A1',
    transparent=True,
)

plt.show()
