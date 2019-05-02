from common import *

N_k = 1 << 8

eb_cou = 0.193
err_eb_cou = 0.005

m_e, m_h, eps_r, T = 0.12, 0.3, 4.90185, 293  # K
sys = system_data(m_e, m_h, eps_r, T)

eps_r = sys.c_aEM * sqrt(2 * sys.m_p / eb_cou)
sys = system_data(m_e, m_h, eps_r, T)

surf_area = 326.4  # nm^2

def diffusion_cx(w, ax, ay, D):
    sqrt_factor_x = sqrt(-1j * w / D) * ax
    sqrt_factor_y = sqrt_factor_x / ax * ay
    return D * (1.0 + tan(-0.5 * sqrt_factor_x) / sqrt_factor_x +
                tan(-0.5 * sqrt_factor_y) / sqrt_factor_y)

exp_power_data = loadtxt('extra/ef_power_spectrum.txt')

w_vec = 2 * pi * exp_power_data[1:, 0]
power_norm_vec = exp_power_data[1:, 1] / simps(exp_power_data[1:, 1], w_vec)

w_peak = w_vec[power_norm_vec.argmax()]  # angular frequency, s^-1
print('w_peak: %e' % w_peak)
ax, ay = 34.0, 10.0  # nm

mu_dc_e = 720  # cm^2 v^-1 s^-1
mu_dc_h = 75  # cm^2 v^-1 s^-1

# d = mu / beta / e
diff_factor = 1e14 / sys.beta

d_e = mu_dc_e * diff_factor  # nm^2 s^-1
d_h = mu_dc_h * diff_factor  # nm^2 s^-1

print('de: %e, dh: %e' % (d_e, d_h))
print('de / w_peak: %f, dh / w_peak: %f' % (d_e / w_peak, d_h / w_peak))

mob_e_vec = diffusion_cx(w_vec, ax, ay, d_e) / diff_factor
mob_h_vec = diffusion_cx(w_vec, ax, ay, d_h) / diff_factor

mob_vec = (mob_e_vec + mob_h_vec)
mob_norm_vec = mob_vec * power_norm_vec

mob = simps(mob_norm_vec, w_vec)
print(mob)

#plt.loglog(w_vec, power_norm_vec, 'b:')
plt.loglog(w_vec, real(mob_e_vec), 'r-', label=r'$\mu_{R,e}(\omega)$')
plt.loglog(w_vec, imag(mob_e_vec), 'r--', label=r'$\mu_{I,e}(\omega)$')
plt.loglog(w_vec, real(mob_h_vec), 'b-', label=r'$\mu_{R,h}(\omega)$')
plt.loglog(w_vec, imag(mob_h_vec), 'b--', label=r'$\mu_{I,h}(\omega)$')
plt.loglog(w_vec, real(mob_vec), 'g-', label=r'$\mu_{R}(\omega)$')
plt.loglog(w_vec, imag(mob_vec), 'g--', label=r'$\mu_{I}(\omega)$')

plt.legend(loc=0)

plt.show()
