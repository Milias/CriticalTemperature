from common import *


def lorentz_cont(energy, gamma_c):
    return 0.5 + arctan(2 * energy / gamma_c) / pi


def model_abs(loaded_data, sizes_vec, hwhm_vec, sys_hh, sys_lh):
    def model_abs_func(xdata, *popt):
        if len(popt) == 23:
            # args:
            # gamma_hh, gamma_lh, peak_hh (4), peak_lh (4)
            # gamma_c, energy_c (4)
            # mag_peak_hh (4), mag_peak_lh (4), mag_cont (4)
            gamma_hh, gamma_lh = popt[:2]
            peak_hh_vec = array(popt[2:6])
            peak_lh_vec = array(popt[6:10])
            gamma_c, energy_c = popt[10], array(popt[11:15])
            mag_peak_lh_vec = popt[15:19]
            mag_cont_vec = popt[19:23]
        elif len(popt) == 20:
            # args:
            # gamma_hh, gamma_lh, peak_hh (4), peak_lh (4)
            # gamma_c, energy_c
            # mag_peak_hh (4), mag_peak_lh (4), mag_cont (4)
            gamma_hh, gamma_lh = popt[:2]
            peak_hh_vec = array(popt[2:6])
            peak_lh_vec = array(popt[6:10])
            gamma_c, energy_c = popt[10], array([popt[11]] * 4)
            mag_peak_lh_vec = popt[12:16]
            mag_cont_vec = popt[16:20]

        print('Γ_hh: %.1f meV, Γ_lh: %.1f meV' % (
            gamma_hh * 1e3,
            gamma_lh * 1e3,
        ))
        print('peak_hh: %.1f, %.1f, %.1f, %.1f meV' % tuple(peak_hh_vec * 1e3))
        print('peak_lh: %.0f, %.0f, %.0f, %.0f meV' % tuple(peak_lh_vec * 1e3))

        print('Γ_c: %.0f meV, ɛ_c: %.1f, %.1f, %.1f, %.1f meV' % (
            gamma_c * 1e3,
            *tuple(energy_c * 1e3),
        ))

        print('mag_peak_lh: %.2f, %.2f, %.2f, %.2f' % tuple(mag_peak_lh_vec))
        print(
            'mag_cont: %.2f, %.2f, %.2f, %.2f\n' % tuple(mag_cont_vec),
            flush=True,
        )

        sum_model_all = []

        for ii, (peak_hh, peak_lh, mag_peak_lh, mag_cont) in enumerate(
                zip(
                    peak_hh_vec,
                    peak_lh_vec,
                    mag_peak_lh_vec,
                    mag_cont_vec,
                )):
            sys_hh.size_Lx, sys_hh.size_Ly = sizes_vec[ii]
            sys_hh.set_hwhm(*hwhm_vec[ii])

            sys_lh.size_Lx, sys_lh.size_Ly = sizes_vec[ii]
            sys_lh.set_hwhm(*hwhm_vec[ii])

            xdata_ii = loaded_data[ii][:, 0]

            data_hh_sum = sum(
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

            data_lh_sum = sum(
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

            data_cont = lorentz_cont(
                xdata_ii - energy_c[ii],
                gamma_c,
            ) * mag_cont

            data_hh_sum /= amax(data_hh_sum)
            data_lh_sum /= amax(data_lh_sum) / mag_peak_lh

            sum_model = data_hh_sum + data_lh_sum + data_cont
            data_hh_sum /= amax(sum_model)
            data_lh_sum /= amax(sum_model)
            data_cont /= amax(sum_model)
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
m_e, m_lh, m_hh, T = 0.27, 0.45, 0.52, 294  # K

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

E_min_data, E_max_data = 0.1, 0.6

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
    0.033934605707915656,
    0.14906456357393272,
    #
    -0.024097954268070147,
    -0.015130346523020833,
    -0.00994487162806537,
    -0.0039029660600132647,
    #
    0.1268478629891975,
    0.1395958892431701,
    0.14601552987793723,
    0.1539140199483991,
    #
    0.17198445886538827,
    #
    0.3121724450324092,
    0.3316654689401301,
    0.33410525949761,
    0.34502708456991943,
    #
    0.762711167962814,
    0.73658886882974,
    0.7368902014698933,
    0.8576454382087552,
    #
    0.7867868185920276,
    0.7223011926246569,
    0.7151897800864765,
    0.8186025998672323,
)

lower_bounds = (
    0,
    0,
    *tuple([-inf] * len(sizes_vec) * 2),
    0,
    *tuple([-inf] * len(sizes_vec)),
    *tuple([0] * len(sizes_vec) * 2),
)

p0_values = (
    0.033941445155198924,
    0.14879979285408973,
    #
    -0.02406494387257569,
    -0.015129243662204247,
    -0.009945865489807599,
    -0.0039301109051036644,
    #
    0.12876658187424717,
    0.1394291067371435,
    0.14563561950633833,
    0.15258302688096678,
    #
    0.17294622931186784,
    #
    0.33024349992812513,
    #
    0.7794379835280983,
    0.7350409091783106,
    0.7335698160508586,
    0.8447021684803576,
    #
    0.8122863233202237,
    0.7212263937077278,
    0.7107648904191728,
    0.7964443782143865,
)

lower_bounds = (
    0,
    0,
    *tuple([-inf] * len(sizes_vec) * 2),
    0,
    -inf,
    *tuple([0] * len(sizes_vec) * 2),
)

popt, pcov = time_func(
    curve_fit,
    model_abs(loaded_data, sizes_vec, hwhm_vec, sys_hh, sys_lh),
    concat_data[:, 0],
    concat_data[:, 1],
    p0=p0_values,
    bounds=(lower_bounds, inf),
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
