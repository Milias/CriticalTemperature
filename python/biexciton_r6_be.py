from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = -193e-3  # eV
eb_biexc_exp = -45e-3  # eV
r_max = 28

E_min = -10000e-3
N_x = 1 << 8
r_min_vec = linspace(1, 10, N_x)

eb_biexc_vec = array(biexciton_be_r6_vec(E_min, eb_cou, r_min_vec, sys))
rmin_exp = biexciton_rmin_r6(E_min, eb_cou, eb_biexc_exp, sys)

plt.title(rmin_exp)

t0 = time.time()
wf_vec = array(biexciton_wf_r6(
    eb_biexc_exp,
    eb_cou,
    rmin_exp,
    r_max,
    N_x,
    sys,
)).reshape(-1, 3)
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_x, dt))

x_vec = linspace(rmin_exp, r_max, N_x)
pot_vec = array(biexciton_pot_r6_vec(eb_cou, x_vec, sys))

plt.axhline(y=0, color='k', linewidth=0.3)
plt.plot(r_min_vec, eb_biexc_vec, 'g-')
plt.plot([rmin_exp], [eb_biexc_exp], 'go')

plt.axvline(x=rmin_exp, color='k', linewidth=0.2)
plt.plot(wf_vec[:, 2], wf_vec[:, 0], 'r-', label=r'$\psi(r)$')
plt.plot(wf_vec[:, 2], wf_vec[:, 1], 'r--', label='$\psi\'(r)$')
plt.plot(x_vec, pot_vec, 'b-', label=r'$1/r^6$')

plt.ylim(-0.5, 1.0)
plt.xlim(0.7 * rmin_exp, r_max)

plt.legend(loc=0)

plt.tight_layout()

plt.show()
