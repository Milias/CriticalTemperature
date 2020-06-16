from common import *


def load_raw_Abs_data(path, eV_min, eV_max):
    raw = loadtxt(path)
    arg_max = raw[:, 1].argmax()
    xdata_eV = raw[:, 0]
    peak_eV = xdata_eV[arg_max]

    xdata_eV_arg = ((xdata_eV - peak_eV) > -eV_min) * (
        (xdata_eV - peak_eV) < eV_max)

    return array([
        xdata_eV[xdata_eV_arg],
        raw[xdata_eV_arg, 1] / amax(raw[xdata_eV_arg, 1]),
    ]).T


file_id_PL = 'd-fccRMqSQKcwq3ju3MHtw'
params_PL_dict = {}
popt_PL = load_data(
    'extra/extcharge/cm_be_polar_fit_params_%s' % file_id_PL,
    params_PL_dict,
)

globals().update(params_PL_dict)

nmax = 5

states_vec = list(
    itertools.product(
        range(1, nmax + 1, 2),
        range(1, nmax + 1, 2),
    ))

N_states = len(states_vec)
N_samples = len(sizes_vec)

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_lh, m_hh, T = 0.27, 0.52, 0.45, 294  # K

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)

E_min_data, E_max_data = 0.1, 0.65

loaded_data = array([
    load_raw_Abs_data(
        'extra/data/extcharge/Abs_%s.txt' % label,
        E_min_data,
        E_max_data,
    ) for label in labels_vec
])

peak_eV_vec = [d[d[:, 1].argmax(), 0] for d in loaded_data]

concat_data = concatenate(loaded_data[:])


def cou_energy(sys):
    return -sys.c_aEM * sys.c_hbarc / sqrt(sys.size_Lx**2 +
                                           sys.size_Ly**2) / sys.eps_mat


def lorentz_cont(energy, gamma_c, sys):
    return array(exciton_cont_vec(energy, gamma_c, sys))
    #return 0.5 + arctan(2 * energy / gamma_c) / pi


def model_abs(loaded_data, sizes_vec, hwhm_vec, sys_hh, sys_lh):
    def model_abs_func(xdata, *popt):
        #args : (20)
        #mag_peak_lh(4), mag_cont_lh(4), mag_cont_hh(4)
        #E_hh[ <0], E_lh[ <0], energy_c_hh[> 0], energy_c_lh[> 0]
        #gamma_lh, gamma_c_hh, gamma_c_lh
        mag_peak_lh_vec = popt[:4]
        mag_cont_lh_vec = popt[4:8]
        mag_cont_hh_vec = popt[8:12]
        E_hh, E_lh = array(popt[12:16]), array(popt[16:20])
        energy_c_hh, energy_c_lh = array(popt[20:24]), array(popt[24:28])
        gamma_hh, gamma_lh, gamma_c_hh, gamma_c_lh = popt_PL[0], *popt[28:31]

        print(time.strftime('%X'))

        print('Γ_hh: %.1f meV, Γ_lh: %.1f meV' % (
            gamma_hh * 1e3,
            gamma_lh * 1e3,
        ))
        print('E_hh: %.1f, %.1f, %.1f, %.1f meV' % tuple(E_hh * 1e3))
        print('E_lh: %.1f, %.1f, %.1f, %.1f meV' % tuple(E_lh * 1e3))
        print('Γ_c_hh: %.0f meV, ɛ_c_hh: %.0f, %.0f, %.0f, %.0f meV' % (
            gamma_c_hh * 1e3,
            *tuple(energy_c_hh * 1e3),
        ))
        print('Γ_c_lh: %.0f meV, ɛ_c_lh: %.0f, %.0f, %.0f, %.0f meV' % (
            gamma_c_lh * 1e3,
            *tuple(energy_c_lh * 1e3),
        ))
        print('mag_peak_lh: %.2f, %.2f, %.2f, %.2f' % mag_peak_lh_vec)
        print('mag_cont_lh: %.2f, %.2f, %.2f, %.2f' % mag_cont_lh_vec)
        print('mag_cont_hh: %.2f, %.2f, %.2f, %.2f' % mag_cont_hh_vec)
        print('\n', flush=True)

        sum_model_all = []

        for ii in range(N_samples):
            sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
            sys_hh.set_hwhm(*hwhm_vec[ii])

            sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
            sys_lh.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            data_cont_hh = lorentz_cont(
                xdata_ii - energy_c_hh[ii],
                gamma_c_hh,
                sys_hh,
            ) * mag_cont_hh_vec[ii]

            data_cont_lh = lorentz_cont(
                xdata_ii - energy_c_lh[ii],
                gamma_c_lh,
                sys_lh,
            ) * mag_cont_lh_vec[ii]

            data_hh_sum = sum(
                array([
                    exciton_lorentz_nomb_vec(
                        xdata_ii - (energy_c_hh[ii] + E_hh[ii]),
                        gamma_hh,
                        nx,
                        ny,
                        sys_hh,
                    ) for nx, ny in states_vec
                ]),
                axis=0,
            )

            data_lh_sum = sum(
                array([
                    exciton_lorentz_nomb_vec(
                        xdata_ii - (energy_c_lh[ii] + E_lh[ii]),
                        gamma_lh,
                        nx,
                        ny,
                        sys_lh,
                    ) for nx, ny in states_vec
                ]),
                axis=0,
            )

            data_hh_sum /= amax(data_hh_sum)
            data_lh_sum /= amax(data_lh_sum) / mag_peak_lh_vec[ii]

            sum_model = data_hh_sum + data_lh_sum + data_cont_hh + data_cont_lh

            sum_model_all.extend(sum_model / amax(sum_model))

        return array(sum_model_all)

    return model_abs_func


#args : (25)
#mag_peak_lh(4), mag_cont_lh(4), mag_cont_hh(4)
#E_hh[ <0](4), E_lh[ <0](4), energy_c_hh[> 0], energy_c_lh[> 0]
#gamma_lh, gamma_c_hh, gamma_c_lh
p0_values = (
    #
    0.5,
    0.5,
    0.5,
    0.5,
    #
    0.3,
    0.3,
    0.3,
    0.4,
    #
    0.4,
    0.3,
    0.3,
    0.3,
    #
    -170e-3,
    -170e-3,
    -170e-3,
    -170e-3,
    #
    -270e-3,
    -270e-3,
    -270e-3,
    -270e-3,
    #
    2.7,
    2.7,
    2.7,
    2.7,
    #
    2.85,
    2.85,
    2.85,
    2.85,
    #
    100e-3,
    20e-3,
    60e-3,
)

lower_bounds = (
    *tuple([0] * N_samples * 3),
    #
    *tuple([-200e-3] * N_samples),
    #
    *tuple([-300e-3] * N_samples),
    #
    0,
    0,
    0,
    0,
    #
    0,
    0,
    0,
    0,
    #
    0,
    0,
    0,
)

upper_bounds = (
    *tuple([1] * N_samples * 3),
    #
    *tuple([-150e-3] * N_samples),
    #
    *tuple([-250e-3] * N_samples),
    #
    inf,
    inf,
    inf,
    inf,
    #
    inf,
    inf,
    inf,
    inf,
    #
    inf,
    inf,
    inf,
)

popt, pcov = time_func(
    curve_fit,
    model_abs(loaded_data, sizes_vec, hwhm_vec, sys_hh, sys_lh),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=p0_values,
    bounds=(lower_bounds, upper_bounds),
    method='trf',
)

print(popt.tolist(), flush=True)

file_id = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode()[:-2]

save_data(
    'extra/extcharge/cm_be_polar_fit_params_abs_%s' % file_id,
    [popt],
    extra_data={
        'labels_vec': labels_vec,
        'sizes_vec': sizes_vec,
        'hwhm_vec': hwhm_vec,
        'states_vec': states_vec,
        'pcov': pcov.tolist(),
    },
)
