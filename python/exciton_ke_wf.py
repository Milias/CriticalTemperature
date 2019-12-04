from common import *


def wf_func(x, N, a0, b):
    return N * sqrt(x) * exp(-x / a0 * 2) * special.hyp1f1(b, 1, x / a0 * 4)


def root_be_cou(eps_r, m_e, m_h, T, be_exc):
    return exciton_be_cou(system_data(m_e, m_h, eps_r, T)) - be_exc


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

N_x = 1 << 12

N_fit_cou = 800
N_fit_ke = 1500

r_max = 28
x_vec = linspace(0.0, r_max, N_x)

be_exc = -193e-3  # eV
m_e, m_h, eps_r, T = 0.22, 0.41, 6.36, 294  # K

size_d = 1.37  # nm
#eps = eps_r
eps = 6.0
eps_sol = 1.9955

sys = system_data(m_e, m_h, eps_r, T)
"""
sys_ke = system_data(m_e, m_h, eps_sol, T)
sys_ke.size_d = size_d
sys_ke.eps_mat = eps_r
set_printoptions(threshold=pysys.maxsize)
#wf_cou_vec = array(exciton_wf_cou(be_exc, r_max, N_x, sys)).reshape(-1, 3)
wf_cou_vec = array(exciton_wf_ke(be_exc, size_d, eps, r_max, N_x,
                                 sys_ke)).reshape(-1, 3)
print(wf_cou_vec)

ax[0].plot(wf_cou_vec[:, 2], wf_cou_vec[:, 0], 'r-', label=r'$\psi(r)$')
ax[0].plot(wf_cou_vec[:, 2], wf_cou_vec[:, 1], 'r--', label='$\psi\'(r)$')

ax[0].set_ylim(-1, 1)
plt.show()
exit()
"""

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / abs(be_exc))
sys = system_data(m_e, m_h, eps_r, T)
print('eps_r(Th): %.5f' % sys.eps_r)

#"""
eps_r = root_scalar(
    root_be_cou,
    args=(m_e, m_h, T, be_exc),
    bracket=[1.0, 10.0],
    method='brentq',
).root
print('eps_r: %.5f' % eps_r)
sys = system_data(m_e, m_h, eps_r, T)
#"""

wf_cou_vec = array(exciton_wf_cou(be_exc, r_max, N_x, sys)).reshape(-1, 3)
pot_cou_vec = array(exciton_pot_cou_vec(wf_cou_vec[:, 2], sys))

try:
    popt, _ = curve_fit(wf_func,
                        wf_cou_vec[:N_fit_cou, 2],
                        wf_cou_vec[:N_fit_cou, 0],
                        p0=(1e6, 1.0, 0.01))
except:
    popt = (1, 1.0, 0.01)

th_pol_cou = 21 / 2**7 * sys.eps_r * popt[1]**3
print('[Coulomb] a0: %f nm' % popt[1])
print('[Coulomb] a0(Th): %f nm' % sys.a0)
print('[Coulomb] b: %f' % popt[2])
print('[Coulomb] th_pol: %f nm^3' % th_pol_cou)

ax[0].plot(x_vec, wf_func(x_vec, *popt) / popt[0], 'r:')

be_exc = exciton_be_cou(sys)

ax[0].axvline(x=popt[1], color='r', linewidth=0.3)

ax[0].plot(wf_cou_vec[:, 2],
           wf_cou_vec[:, 0] / popt[0],
           'r-',
           label=r'$\psi(r)$')
"""
ax[0].plot(wf_cou_vec[:, 2],
           wf_cou_vec[:, 1] / popt[0],
           'r--',
           label='$\psi\'(r)$')
"""

ax[0].plot(wf_cou_vec[:, 2],
           pot_cou_vec,
           'r-',
           linewidth=0.8,
           label=r'Coulomb')

eps = root_scalar(
    lambda eps: exciton_be_ke(size_d, eps, system_data(m_e, m_h, eps_sol, T)) -
    be_exc,
    bracket=[2.0, 10.0],
    method='brentq',
).root
sys = system_data(m_e, m_h, eps_sol, T)
sys.size_d = size_d
sys.eps_mat = eps

be_exc_ke = exciton_be_ke(size_d, eps, sys)
print('[Compare] E_B(Cou): %.0f meV, E_B(Keldysh): %.0f meV' %
      (1e3 * be_exc, 1e3 * be_exc_ke))

print('[Keldysh] solution: %.3f, material: %.3f' % (eps_sol, eps))

wf_ke_vec = array(exciton_wf_ke(be_exc_ke, size_d, eps, r_max, N_x,
                                sys)).reshape(-1, 3)

pot_ke_vec = array(exciton_pot_ke_vec(size_d, eps, wf_ke_vec[:, 2], sys))

try:
    popt, _ = curve_fit(
        wf_func,
        wf_ke_vec[:N_fit_ke, 2],
        wf_ke_vec[:N_fit_ke, 0],
        p0=(1, 1.0, 0.01),
    )
except:
    popt = (1e6, 1.0, 0.01)

th_pol_ke = 21 / 2**7 * sys.eps_r * popt[1]**3
print('[Keldysh] a0: %f nm' % popt[1])
print('[Keldysh] b: %f' % popt[2])
print('[Keldysh] th_pol: %f nm^3' % th_pol_ke)

ax[0].set_title(
    '$\epsilon$: $%.3f$ $\epsilon_{sol}$: $%.3f$\n$a^{(Cou)}$: $%.2f$ nm$^3$, $a^{(Kel)}$: $%.2f$ nm$^3$, Diff: %d\%%'
    % (
        eps,
        sys.eps_r,
        th_pol_cou,
        th_pol_ke,
        200 * abs(th_pol_cou - th_pol_ke) / (th_pol_cou + th_pol_ke),
    ))

ax[0].axvline(x=wf_cou_vec[N_fit_cou, 2], color='r', linestyle='--')
ax[0].axvline(x=wf_ke_vec[N_fit_ke, 2], color='b', linestyle='--')

ax[0].axvline(x=popt[1], color='b', linewidth=0.3)
ax[0].plot(x_vec, wf_func(x_vec, *popt) / popt[0], 'b:')

ax[0].axhline(y=0, color='k', linewidth=0.3)
ax[0].axhline(y=be_exc, color='g', linewidth=0.6)

ax[0].plot(wf_ke_vec[:, 2], wf_ke_vec[:, 0] / popt[0], 'b-')
#ax[0].plot(wf_ke_vec[:, 2], wf_ke_vec[:, 1] / popt[0], 'b--')

ax[0].plot(wf_ke_vec[:, 2], pot_ke_vec, 'b-', linewidth=0.8, label=r'Keldysh')

ax[0].set_ylim(-0.5, 1)
ax[0].set_xlim(0.7 * wf_cou_vec[0, 2], r_max)

ax[0].legend(loc=0)

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Excitons/%s.pdf' %
            'exciton_wf_ke')

plt.show()
