from common import *

plt.rcParams.update({'font.size': 16})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'text.usetex': True,
})

fig_size = tuple(array([6.8, 5.3]) * 2)

n_x, n_y = 1, 1
fig = plt.figure(figsize=fig_size)
ax = [fig.add_subplot(n_x, n_y, i + 1) for i in range(n_x * n_y)]

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

be_cou = -193e-3  # eV
be_biexc_exp = -45e-3  # eV
r_max = 28

N_x = 1 << 10

param_c12 = biexciton_c12_lj(be_cou, be_biexc_exp, sys)
#param_c12 = 3.0

be_cou_scr = -180e-3  # eV
be_biexc_scr = biexciton_be_lj(param_c12, be_cou_scr, sys)

ax[0].set_title(
    '$C_{12}$: %.2f eV nm$^{12}$\n$C_6^{(exp)}$: %.2f eV nm$^{6}$, $C_6^{(scr)}$: %.2f eV nm$^{6}$\n$E_{X_2}^{(exp)}$: $%d$ meV, $E_{X_2}^{(scr)}$: $%d$ meV'
    % (
        param_c12,
        biexciton_lj_c6(be_cou, sys),
        biexciton_lj_c6(be_cou_scr, sys),
        1e3 * be_biexc_exp,
        1e3 * be_biexc_scr,
    ))

t0 = time.time()
wf_vec = array(
    biexciton_wf_lj(
        be_biexc_scr,
        be_cou_scr,
        param_c12,
        r_max,
        N_x,
        sys,
    )).reshape(-1, 3)
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_x, dt))

pot_vec = array(biexciton_pot_lj_vec(param_c12, be_cou, r_max, N_x, sys))
pot_vec_scr = array(
    biexciton_pot_lj_vec(param_c12, be_cou_scr, r_max, N_x, sys))

ax[0].axhline(y=0, color='k', linewidth=0.3)
ax[0].axhline(y=be_biexc_exp, color='g', linewidth=0.3)
ax[0].axhline(y=be_biexc_scr, color='b', linewidth=0.5)

ax[0].axvline(x=wf_vec[0, 2], color='k', linewidth=0.2)

ax[0].plot(wf_vec[:, 2], wf_vec[:, 0], 'r-', label=r'$\psi(r)$')
ax[0].plot(wf_vec[:, 2], wf_vec[:, 1], 'r--', label='$\psi\'(r)$')

ax[0].plot(wf_vec[:-1, 2], pot_vec, 'g-', linewidth=0.8, label=r'LJ')
ax[0].plot(wf_vec[:-1, 2], pot_vec_scr, 'b-', label=r'LJ Screened')

ax[0].set_ylim(1.3 * amin(pot_vec_scr), -2 * amin(pot_vec_scr))
ax[0].set_xlim(0.7 * wf_vec[0, 2], r_max)

ax[0].legend(loc=0)

plt.tight_layout()

plt.savefig('/storage/Reference/Work/University/PhD/Biexcitons/%s.png' % 'biexciton_lj_wf', dpi = 600)
plt.savefig('/storage/Reference/Work/University/PhD/Biexcitons/%s.pdf' % 'biexciton_lj_wf')

plt.show()
