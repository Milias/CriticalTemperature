from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
#m_e, m_h, eps_r, T = 1, 1, 1, 1  # K
sys = system_data(m_e, m_h, eps_r, T)
sys_ls = 0.5 / pi * sys.c_aEM / sys.eps_r * (sys.m_e + sys.m_h) / sys.c_hbarc

N_k = 1 << 9
N_ls = 12

N_u, N_v = N_k, N_k
k0, k1, u_max = 2.0, 0, 1

k, dk = linspace(0, k0, N_k, retstep=True)
kk_vec = list(itertools.product(k, repeat=2))

u, du = linspace(u_max / N_u, u_max, N_u, retstep=True)
vu_vec = list(itertools.product(u, repeat=2))

r_k = range(N_k)
id_vec = list(itertools.product(r_k, repeat=2))
id_vu_vec = list(itertools.product(r_k, repeat=2))

z_cou_wf = wf_2d_E_cou_py(sys)
z_sys_wf = wf_2d_E_lim_py(sys_ls, sys)

z_cou = plasmon_det_zero_vu_simp_lwl(vu_vec, id_vu_vec, N_k, du, 1e-8, sys)
z_sys = plasmon_det_zero_vu_simp_lwl(vu_vec, id_vu_vec, N_k, du, sys_ls, sys)

print('sys_ls: \t%8.6f nm' % (1 / sys_ls))
print('z_cou_2d:\t%8.6f eV' % sys.get_E_n(0.5))
print(
    'z_cou_wf:\t%8.6f eV, z_sys_wf:\t%8.6f eV\nz_cou:   \t%8.6f eV, z_sys:   \t%8.6f eV'
    % (z_cou_wf, z_sys_wf, z_cou, z_sys))

plt.axhline(y=0, color='k')

#plt.axhline(y=z_cou, color='g', label='(D) $\epsilon_{B,Cou}$')
plt.axhline(
    y=z_cou_wf, color='g', linestyle='--', label='(W) $\epsilon_{B,Cou}$')

plt.axhline(
    y=z_sys, color='m', label='(D) $\epsilon_{B,lwl}(\lambda_{s,0}^{-1})$')
plt.axhline(
    y=z_sys_wf,
    color='m',
    linestyle='--',
    label='(W) $\epsilon_{B,lwl}(\lambda_{s,0}^{-1})$')

x_list = logspace(-4, 1, N_ls)

plt.axhline(y=0, color='k')
plt.axhline(y=z_cou_wf, color='g', label='Coulomb binding energy')
plt.axvline(x=sys_ls, color='m', label='$\lambda_{s,0}^{-1}$')

z_list = []
z_wf_list = []
for ls in x_list:
    t0 = time.time()
    z = plasmon_det_zero_vu_simp_lwl(vu_vec, id_vu_vec, N_k, du, ls, sys)

    z_wf = wf_2d_E_lim_py(ls, sys)

    print('[%e] (%f, %f) r: %f, Elapsed: %.2fs' %
          (ls, z, z_wf, 2 * abs(z_wf - z) / abs(z_wf + z), time.time() - t0))

    z_list.append(z)
    z_wf_list.append(z_wf)

z_arr, z_wf_arr = array(z_list), array(z_wf_list)

plt.semilogx(x_list, z_arr, 'r.-', label='Discretization')
plt.semilogx(x_list, z_wf_arr, 'b.-', label='Wavefunction')
#plt.plot(x_list, real(z_wf_arr) / real(z_arr), 'g.--')

plt.title('$\epsilon_B$ vs. $\lambda_s^{-1}$\nLong-wavelength limit')
plt.xlabel('$\lambda_s^{-1}$ / dimensionless')
plt.ylabel('$\epsilon_B$ / dimensionless')

plt.legend(loc=0)

plt.savefig('plots/poles_lwl_dims.eps')

plt.show()
