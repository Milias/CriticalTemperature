from common import *

m_e, m_h, eps_r, T = 0.22, 0.41, 6.369171898453055, 294  # K
T_vec = array([T])
sys = system_data(m_e, m_h, eps_r, T)

eb_cou = 193e-3  # eV

file_id = 'tXV6CHV0SySMglAEgovJow'  # 256 [1e-2, 32] log

data = load_data('extra/biexcitons/eff_pot_vec_%s' % file_id, globals())
param_vec = data[:, 0]
pot_vec = data[:, 1:]

pol_a2 = 21 / 2**8
th_pol_cou = 2 * pol_a2 * eps_r * (2 * sys.c_hbarc**2 / eb_cou / sys.m_p)**1.5
param_c6 = 24 / (eb_cou**2 * sys.m_p**3) * (pol_a2 * eps_r * sys.c_hbarc**3)**2
"""
pot_vec[:, 6] = pot_vec[:, 6] * (1.0 - exp(-param_vec / sys.a0)) - exp(
    -param_vec / sys.a0) * param_c6 / param_vec**6
"""

eb_biexc_cou = 45e-3

eb_biexc_comp = biexciton_be(amin(pot_vec[:, 6]), param_vec, pot_vec[:, 6],
                             sys)

print(eb_biexc_comp)

N_param = 1 << 10
t0 = time.time()
wf_vec = array(
    biexciton_wf(eb_biexc_comp, param_vec, pot_vec[:, 6], N_param,
                 sys)).reshape(-1, 3)
dt = (time.time() - t0)
print('dt/N: %.3e s, dt: %.3f s' % (dt / N_param, dt))

x_interp_vec = linspace(param_vec[0], param_vec[-1], N_param)
pot_interp_vec = array([
    result_s(r).total_value() for r in biexciton_eff_pot_interp_vec(
        param_vec,
        pot_vec[:, 6],
        x_interp_vec,
        sys,
    )
])

plt.plot(wf_vec[:, 2], wf_vec[:, 0], 'r-')
plt.plot(wf_vec[:, 2], wf_vec[:, 1], 'r--')
plt.plot(x_interp_vec, pot_interp_vec, 'b-', label='$V_-(r_{BA})$ interp')
plt.axhline(y=0, color='k', linewidth=0.3)

plt.ylim(-1, 30)
plt.xlim(0, 28)

plt.show()
