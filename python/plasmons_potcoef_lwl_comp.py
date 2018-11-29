from common import *

N_k = 1 << 12

m_e, m_h, eps_r, T = 0.28, 0.59, 6.56, 1  # K
sys = system_data(m_e, m_h, eps_r, T)

v_1 = 1 / (sys.c_aEM * sqrt(2 * sys.m_pT))
sys.set_v_1(v_1)

sys_ls = 2 / pi * sys.c_aEM / sys.eps_r * sys.m_e / sys.c_hbarc / sys.k_F
z_wf = wf_2d_E_lim_py(0, sys) * sys.e_F

print('z_wf: %f, sys_ls: %e' % (z_wf, sys_ls))
print('E_F: %4.3f eV, k_F: %4.3f nm^-1' % (sys.e_F, sys.k_F * 1e-9))

N_cols, N_rows = 3, 4

x_max, N_x = 10.0, N_cols*N_rows - 1
x_list = logspace(-5, log10(x_max), N_x)

k0 = 10

k_vec = linspace(k0 / N_k, k0, N_k)
kk_prod = list(itertools.product(k_vec, repeat=2))

potcoef_lwl_sys = array(plasmon_potcoef_lwl_v(kk_prod, sys_ls, sys)).reshape(N_k, N_k)[::-1, :]
potcoef_lwl_list = []

for ls in x_list:
    t0 = time.time()

    potcoef_lwl = array(plasmon_potcoef_lwl_v(kk_prod, ls, sys)).reshape(N_k, N_k)[::-1, :]

    print('[%e] Elapsed: %.2fs' % (ls, time.time() - t0))

    potcoef_lwl_list.append(potcoef_lwl)

plt.subplot(N_cols, N_rows, 1)
plt.title('LWL: $\lambda_{sys}^{-1}: %.2f$' % sys_ls)

plt.imshow(
    clip(potcoef_lwl_sys, 0, 1),
    cmap=cm.hot,
    aspect='auto',
    extent=(k0 / N_k, k0, k0 / N_k, k0)
    #norm=SymLogNorm(linthresh=1e-2 * amax(zi))
)

for ii, ls in enumerate(x_list):
    plt.subplot(N_cols, N_rows, ii + 2)
    if ls > 1e-2:
        plt.title('LWL: $\lambda_s^{-1}: %.2f$' % ls)
    else:
        plt.title('LWL: $\lambda_s^{-1}: %.2e$' % ls)

    plt.imshow(
        clip(potcoef_lwl_list[ii], 0, 1),
        cmap=cm.hot,
        aspect='auto',
        extent=(k0 / N_k, k0, k0 / N_k, k0)
        #norm=SymLogNorm(linthresh=1e-2 * amax(zi))
    )

#plt.title()
#plt.xlabel()
#plt.ylabel()

#plt.legend(loc=0)

plt.savefig('plots/potcoef_comp_lwl.eps')

plt.show()
