from common import *

N_k = 1 << 8

Lx, Ly, Lz = 34.0, 10.0, 1.4  # nm
L_vec = array([Lx, Ly])
surf_area = 326.4  # nm^2
#surf_area = Lx * Ly  # nm^2

file_id = 'aneuiPMlRLy4x8FlcAajaA'
load_data('extra/mu_e_data_%s' % file_id, globals())
"""
eps_r = 1
m_e = 1
m_h = 2e3
"""

sys = system_data(m_e, m_h, eps_r, T)

print(sys.get_E_n(0.5))

z_cou_lwl = time_func(
    plasmon_det_zero_lwl,
    N_k,
    1e-4,
    sys,
    -1e-2,
)

z_ht = time_func(
    plasmon_det_zero_ht_v,
    N_k,
    array([-1e1]),
    sys,
    -1e-3,
)

print(z_ht)
print(z_cou_lwl)
