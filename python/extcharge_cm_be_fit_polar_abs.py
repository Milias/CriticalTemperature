from common import *


def lorentz_cont(energy, gamma_c):
    return 0.5 + arctan(2 * energy / gamma_c) / pi


def model_abs(loaded_data, sizes_vec, hwhm_vec, sys_hh, sys_lh):
    def model_abs_func(xdata, *args):
        # args:
        # gamma_hh, gamma_lh, peak_hh (4), peak_lh (4), gamma_c, energy_c, mag_peak_lh (4), mag_cont (4)
        gamma_hh, gamma_lh = args[:2]
        peak_hh_vec = array(args[2:6])
        peak_lh_vec = array(args[6:10])
        gamma_c, energy_c = args[10:12]
        mag_peak_lh_vec = args[12:16]
        mag_cont_vec = args[16:22]

        print('Γ_hh: %.2f meV, Γ_lh: %.2f meV' % (
            gamma_hh * 1e3,
            gamma_lh * 1e3,
        ))
        print('peak_hh: %.2f, %.2f, %.2f, %.2f meV' % tuple(peak_hh_vec * 1e3))
        print('peak_lh: %.2f, %.2f, %.2f, %.2f meV' % tuple(peak_lh_vec * 1e3))

        print('Γ_c: %.2f meV, ɛ_c: %.2f meV' % (
            gamma_c * 1e3,
            energy_c * 1e3,
        ))

        print(
            'mag_peak: %.1f, %.1f, %.1f, %.1f\nmag_cont: %.1f, %.1f, %.1f, %.1f\n'
            % (mag_peak_lh, mag_cont),
            flush=True,
        )

        sum_model_all = []

        for ii, (peak_hh, peak_lh) in enumerate(zip(peak_hh_vec, peak_lh_vec)):
            sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
            sys_hh.set_hwhm(*hwhm_vec[ii])

            sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
            sys_lh.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            data_hh = sum(
                array([
                    exciton_PL_vec(
                        xdata_ii - peak_hh,
                        gamma_hh,
                        nx,
                        ny,
                        sys_hh,
                    ) for nx, ny in states_vec
                ]),
                axis=0,
            )

            data_lh = sum(
                array([
                    exciton_PL_vec(
                        xdata_ii - peak_lh,
                        gamma_lh,
                        nx,
                        ny,
                        sys_lh,
                    ) for nx, ny in states_vec
                ]),
                axis=0,
            )

            data_cont = lorentz_cont(xdata_ii - energy_c, gamma_c)

            sum_model = data_hh / amax(data_hh) + mag_peak_lh * data_lh / amax(
                data_lh) + mag_cont * data_cont
            sum_model /= amax(sum_model)

            sum_model_all.extend(sum_model)

        return array(sum_model_all)

    return model_abs_func


def load_raw_Abs_data(path, eV_min, eV_max):
    raw = loadtxt(path)
    arg_max = raw[:, 1].argmax()
    xdata_eV = raw[:, 0]
    xdata_eV -= xdata_eV[arg_max]

    xdata_eV_arg = (xdata_eV > -eV_min) * (xdata_eV < eV_max)

    return array([
        xdata_eV[xdata_eV_arg],
        raw[xdata_eV_arg, 1] / amax(raw[xdata_eV_arg, 1]),
    ]).T


nmax = 5

states_vec = list(
    itertools.product(
        range(1, nmax + 1, 2),
        range(1, nmax + 1, 2),
    ))

N_states = len(states_vec)

size_d = 1.37  # nm
eps_sol = 6.8981
m_e, m_hh, m_lh, T = 0.27, 0.45, 0.52, 294  # K

labels_vec = [
    'BS065',
    'BS006',
    'BS066',
    'BS068',
]
sizes_vec = [
    (29.32, 5.43),
    (26.11, 6.42),
    (25.4, 8.05),
    (13.74, 13.37),
]
hwhm_vec = [
    (3.3, 0.81),
    (3.34, 1.14),
    (2.9, 0.95),
    (2.17, 1.85),
]

sys_hh = system_data(m_e, m_hh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)
sys_lh = system_data(m_e, m_lh, eps_sol, T, size_d, 0, 0, 0, 0, eps_sol)

E_min_data, E_max_data = 0.15, 0.7

loaded_data = array([
    load_raw_Abs_data(
        'extra/data/extcharge/Abs_%s.txt' % label,
        E_min_data,
        E_max_data,
    ) for label in labels_vec
])

concat_data = concatenate(loaded_data[:])

popt_list, pcov_list = [], []

p0_values = (
    40e-3,
    120e-3,
    -0.03,
    -0.02,
    -0.01,
    0.00,
    0.12,
    0.135,
    0.145,
    0.155,
    80e-3,
    0.43,
    0.75,
    0.75,
    0.75,
    0.75,
    0.7,
    0.7,
    0.7,
    0.7,
)

popt, pcov = time_func(
    curve_fit,
    model_abs(loaded_data, sizes_vec, hwhm_vec, sys_hh, sys_lh),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=p0_values,
    bounds=((
        0,
        0,
        *tuple([-inf] * len(sizes_vec) * 2),
        0,
        -inf,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ), inf),
    method='trf',
)

print(popt)
print(pcov)

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
        'n_gamma': len(gamma_p0),
    },
)
