from common import *

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

#e_F = 1  # eV
#sys.set_energy_fermi(e_F)

#sys.set_v_1_fermi(1)

sys_ls = 4 * sys.c_aEM / sys.eps_r * sys.m_e / sys.c_hbarc # nm^-1
z_wf = wf_2d_E_lim_py(0, sys)
z_sys_wf = wf_2d_E_lim_py(sys_ls, sys)

print('z_wf: %f, z_sys_wf: %f, sys_ls: %f nm, v_1: %f' % (z_wf, z_sys_wf, 1/sys_ls, sys.v_1))
#print('E0: %3.2e eV, l0: %4.3f nm' % (sys.E0, sys.l0))

x_list = linspace(10, 50, 6)
y_list = array([int(x) for x in (2**linspace(7, 9, 3))])

plt.axhline(y=0, color='k')
#plt.axhline(y=1, color='m')
plt.axhline(y=z_wf, color='g', label='Coulomb binding energy')
plt.axhline(y=z_sys_wf, color='b', label='Wavefunction')

#sys_ls = 1e-4

z_list = []
for N_k in y_list:
    r_k = range(N_k)
    ids_prod = list(itertools.product(r_k, repeat=2))

    check_nan = False

    for k0 in x_list:
        k = linspace(k0 / N_k, k0, N_k)
        kk_prod = list(itertools.product(k, repeat=2))

        dk = k[1] - k[0]

        t0 = time.time()
        if not check_nan:
            z = plasmon_sysmat_det_zero_lwl(kk_prod, ids_prod, dk, int(N_k),
                                            sys_ls, sys.v_1, sys)

            if isnan(z):
                check_nan = True
        else:
            z = float('nan')

        print('[%d,%e] (%f, %f) r: %f, Elapsed: %.2fs' %
              (N_k, k0, z, z_sys_wf, z / z_sys_wf, time.time() - t0))

        z_list.append(z)

z_arr = array(z_list).reshape((y_list.size, x_list.size))

for ii, N_k in enumerate(y_list):
    plt.plot(x_list, real(z_arr[ii]), '.-', label='N_k: %d' % N_k)
    #plt.plot(x_list, real(z_arr[ii]) / z_sys_wf, 'm.--')

plt.title('$\epsilon_B$ vs. $k_0$\nLong-wavelength limit')
plt.xlabel('$k_0$ / dimensionless')
plt.ylabel('$\epsilon_B$ / dimensionless')

plt.legend(loc=0)

plt.savefig('plots/poles_lwl_sys_all.eps')

plt.show()
