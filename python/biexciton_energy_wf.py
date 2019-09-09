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

fig_size = tuple(array([6.8, 5.3]) * 2)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

surf_area = 326.4  # nm^2
m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
eb_cou = 0.193  # eV
eb_biexc_cou = 45e-3  # eV
sys = system_data(m_e, m_h, eps_r, T)

pol_a2 = 21 / 2**8
th_pol_cou = 2 * pol_a2 * eps_r * (2 * sys.c_hbarc**2 / eb_cou / sys.m_p)**1.5
"""
mu_e_lim = plasmon_exc_mu_val(eb_exc + 0.5 * eb_exc2, sys)
mu_exc_lim = mu_e_lim + sys.get_mu_h(mu_e_lim)
print('mu_e_lim: %f, mu_exc_lim: %f' % (mu_e_lim, mu_exc_lim))
"""


def root_func(lj_param_a, eb_biexc_cou, lj_param_b):
    if not isinstance(lj_param_a, double):
        lj_param_a = lj_param_a[0]

    lj_param_a = sys.c_hbarc * exp(lj_param_a)
    return wf_bexc_E(-lj_param_b**2 * 0.25 / lj_param_a, lj_param_a,
                     lj_param_b, sys) + eb_biexc_cou


lj_param_b = 24 / (eb_cou**2 * sys.m_p**3) * (pol_a2 * eps_r *
                                              sys.c_hbarc**3)**2
lj_param_a_sol = root(
    root_func,
    [0.0],
    args=(eb_biexc_cou, lj_param_b),
    method='hybr',
)

lj_param_a = sys.c_hbarc * exp(lj_param_a_sol.x[0])

eb_exc = 1.0 * eb_cou
lj_param_b = 24 / (eb_exc**2 * sys.m_p**3) * (pol_a2 * eps_r *
                                              sys.c_hbarc**3)**2
print("A: %f, B: %f" % (lj_param_a, lj_param_b))

r_m = (2.0 * lj_param_a / lj_param_b)**(1.0 / 6)
r_min = (1.0 + sqrt(2.0))**(-1.0 / 6) * r_m
r_max = 4 * r_m

v_min = lj_param_b**2 * 0.25 / lj_param_a

#"""
bexc_eb = wf_bexc_E(
    -v_min,
    lj_param_a,
    lj_param_b,
    sys,
)
print(bexc_eb)
#"""



#"""
wf_vec = array(wf_bexc_s_r(
    r_max,
    bexc_eb,
    lj_param_a,
    lj_param_b,
    sys,
)).reshape(-1, 3)

print("WF maximum: %f nm" % wf_vec[wf_vec[:, 0].argmax(), 2])

ax[0].axhline(y=0, linewidth=0.3, linestyle='-', color='k')

ax[0].plot(wf_vec[:, 2], wf_vec[:, 0], 'r-')
ax[0].plot(wf_vec[:, 2], wf_vec[:, 1], 'r--')
#"""

#"""
ax[0].plot(
    wf_vec[:, 2],
    lj_param_a / wf_vec[:, 2]**12 - lj_param_b / wf_vec[:, 2]**6,
    'b-',
)
ax[0].axvline(x=lj_param_b)
#"""

ax[0].set_ylim(-1.1 * v_min, 1.0 * v_min)
ax[0].set_xlim(r_min, r_max)

plt.tight_layout()
plt.show()
