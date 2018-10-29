#include "plasmons.h"
#include "plasmons_utils.h"

std::vector<double> plasmon_green(
    double w, double k,
    double mu_e, double mu_h,
    double v_1, const system_data & sys,
    double delta
) {
    std::complex<double> w_complex(w, delta);

    double E[2] = { sys.m_pe * k * k, sys.m_ph * k * k };

    std::complex<double> nu[4] = {
        -w_complex / E[0] - 1.0, -w_complex / E[0] + 1.0,
        -w_complex / E[1] - 1.0, -w_complex / E[1] + 1.0
    };

    std::complex<double> pi_screen_nofactor[2] = {
        -2.0 - nu[0] * std::sqrt(1.0 - 4.0 * mu_e / (E[0] * nu[0] * nu[0])) + nu[1] * std::sqrt(1.0 - 4.0 * mu_e / (E[0] * nu[1] * nu[1])),
        -2.0 - nu[2] * std::sqrt(1.0 - 4.0 * mu_h / (E[1] * nu[2] * nu[2])) + nu[3] * std::sqrt(1.0 - 4.0 * mu_h / (E[1] * nu[3] * nu[3]))
    };

    std::complex<double> green = 1.0 / (-sys.eps_r * v_1 * std::abs(k) +
        (pi_screen_nofactor[0] / sys.m_pe + pi_screen_nofactor[1] / sys.m_ph) * 0.125 * M_1_PI);

    return { green.real(), green.imag() };
}

double plasmon_kmax_f(double k, void * params) {
    plasmon_kmax_s * s{ static_cast<plasmon_kmax_s *>(params) };

    double w{ s->sys.m_pe * k * k + 2.0 * std::sqrt(s->sys.m_pe * s->mu_e) * k };

    double E[2] = { s->sys.m_pe * k * k, s->sys.m_ph * k * k };

    double nu[4] = { -w / E[0] - 1.0, -w / E[0] + 1.0, -w / E[1] - 1.0,
        -w / E[1] + 1.0
    };

    double pi_screen_nofactor[2] = {
        -2.0 - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[0] * nu[0]))
        /*
             * When m_e < m_h, the argument of the sqrt is zero (or very close),
             * so it has to be removed to avoid complex numbers because of
             * numerical precision.
             */
        /*+ nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / ( E[0] * nu[1] * nu[1] ))*/,
        -2.0 - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[2] * nu[2])) + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[3] * nu[3]))
};

    double r{ -s->sys.eps_r * s->v_1 * k + (pi_screen_nofactor[0] / s->sys.m_pe + pi_screen_nofactor[1] / s->sys.m_ph) * 0.125 * M_1_PI };

    return r;
}

double plasmon_kmax(double mu_e, double mu_h, double v_1,
    const system_data & sys) {
    /*
     * Evaluates the Green's function at
     *   w = m_pe * k^2 + 2 * sqrt(mu_e * m_pe) * k
     * and searches for the point at which Ginv is zero.
     *
     * This point is when the plasmon pole disappears.
     * Assumes k > 1e-5.
     * TODO: what about k < 1e-5?
     *
     * First perform an exponential sweep to find the upper
     * bound.
     * TODO: improve this? Analytic upper bound?
     */

    if(v_1 > 1e5) {
        return v_1;
    }

    struct plasmon_kmax_s s {
        mu_e, mu_h, v_1, sys
    };

    double z{ 0 };

    double z_min{ 1e-5 }, z_max{ 1.0 };

    gsl_function funct;
    funct.function = &plasmon_kmax_f;
    funct.params = &s;

    /*
     * Expontential sweep
     */

    const uint32_t max_pow{ 20 };
    double ginv_upper{ 0 }, upper_bound{ z_max };

    for(uint32_t ii = 1; ii <= max_pow; ii++) {
        ginv_upper = funct.function(upper_bound, funct.params);

        if(ginv_upper < 0) {
            z_max = upper_bound;
            break;

        }

        else if(ginv_upper == 0) {
            return z_max;

        }

        else {
            z_min = upper_bound;
            upper_bound = z_max * (1 << ii);
        }
    }

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for(int status = GSL_CONTINUE, iter = 0;
        status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z = gsl_root_fsolver_root(solver);
        z_min = gsl_root_fsolver_x_lower(solver);
        z_max = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_wmax(double mu_e, double mu_h, double v_1,
    const system_data & sys) {
    double kmax{ plasmon_kmax(mu_e, mu_h, v_1, sys) };
    return sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax;
}

double plasmon_wmax(double kmax, double mu_e, const system_data & sys) {
    return sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax;
}

double plasmon_disp_f(double w, void * params) {
    plasmon_disp_s * s{ static_cast<plasmon_disp_s *>(params) };

    double E[2] = { s->sys.m_pe * s->k * s->k, s->sys.m_ph * s->k * s->k };

    double nu[4] = { -w / E[0] - 1.0, -w / E[0] + 1.0, -w / E[1] - 1.0,
            -w / E[1] + 1.0
        };

    double pi_screen_nofactor[2] = {
        -2.0 - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[0] * nu[0])) + nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[1] * nu[1])),
            -2.0 - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[2] * nu[2])) + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[3] * nu[3]))
        };

    double r{ -s->sys.eps_r * s->v_1 * s->k + (pi_screen_nofactor[0] / s->sys.m_pe + pi_screen_nofactor[1] / s->sys.m_ph) * 0.125 * M_1_PI };

    return r;
}

template <bool check_bounds>
double plasmon_disp_tmpl(double k, double mu_e, double mu_h, double v_1,
    const system_data & sys, double kmax = 0) {
    if(k == 0) {
        return 0;

    }

    else if(k < 0) {
        k = -k;
    }

    if constexpr(check_bounds) {
        kmax = plasmon_kmax(mu_e, mu_h, v_1, sys);

        if(k > kmax) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double z{ 0 };
    double z_min{ sys.m_pe * k * k + 2 * std::sqrt(sys.m_pe * mu_e) * k },
           z_max{ sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax };

    z_min = z_min + (z_max - z_min) * 1e-5;

    struct plasmon_disp_s s {
        k, mu_e, mu_h, v_1, sys
    };

    gsl_function funct;
    funct.function = &plasmon_disp_f;
    funct.params = &s;

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for(int status = GSL_CONTINUE, iter = 0;
        status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z = gsl_root_fsolver_root(solver);
        z_min = gsl_root_fsolver_x_lower(solver);
        z_max = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_disp(
    double k, double mu_e, double mu_h, double v_1,
    const system_data & sys
) {
    return plasmon_disp_tmpl<true> (k, mu_e, mu_h, v_1, sys);
}

double plasmon_disp_ncb(
    double k, double mu_e, double mu_h, double v_1,
    const system_data & sys, double kmax
) {
    return plasmon_disp_tmpl<false> (k, mu_e, mu_h, v_1, sys, kmax);
}

double plasmon_disp_inv_f(double k, void * params) {
    plasmon_disp_inv_s * s{ static_cast<plasmon_disp_inv_s *>(params) };

    double E[2] = { s->sys.m_pe * k * k, s->sys.m_ph * k * k };

    double nu[4] = { -s->w / E[0] - 1.0, -s->w / E[0] + 1.0, -s->w / E[1] - 1.0,
            -s->w / E[1] + 1.0
        };

    double pi_screen_nofactor[2] = {
        -2.0 - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[0] * nu[0])) + nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[1] * nu[1])),
        -2.0 - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[2] * nu[2])) + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[3] * nu[3]))
        };

    double r{ -s->sys.eps_r * s->v_1 * k + (pi_screen_nofactor[0] / s->sys.m_pe + pi_screen_nofactor[1] / s->sys.m_ph) * 0.125 * M_1_PI };

    return r;
}

template <bool check_bounds>
double plasmon_disp_inv_tmpl(double w, double mu_e, double mu_h, double v_1,
    const system_data & sys) {
    /*
     * Essentially same as plasmon_disp, but it computes k(w), the inverse
     * dispersion relation.
     */

    if (w == 0) {
        return 0;
    } else if (w < 1e-5) {
        return w;
    } else if (w < 0) {
        w = -w;
    }

    if constexpr(check_bounds) {
        double kmax{ plasmon_kmax(mu_e, mu_h, v_1, sys) };
        double w_max{ sys.m_pe * kmax * kmax + 2 * std::sqrt(sys.m_pe * mu_e) * kmax };

        if(w > w_max) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

    double z{ 0 };
    double z_min{ 1e-5 },
           z_max{ (-std::sqrt(mu_e) + std::sqrt(mu_e + w)) / std::sqrt(sys.m_pe) };

    struct plasmon_disp_inv_s s {
        w, mu_e, mu_h, v_1, sys
    };

    gsl_function funct;
    funct.function = &plasmon_disp_inv_f;
    funct.params = &s;

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for(int status = GSL_CONTINUE, iter = 0;
        status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z = gsl_root_fsolver_root(solver);
        z_min = gsl_root_fsolver_x_lower(solver);
        z_max = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return z;
}

double plasmon_disp_inv(double w, double mu_e, double mu_h, double v_1,
    const system_data & sys) {
    return plasmon_disp_inv_tmpl<true> (w, mu_e, mu_h, v_1, sys);
}

double plasmon_disp_inv_ncb(double w, double mu_e, double mu_h, double v_1,
    const system_data & sys) {
    return plasmon_disp_inv_tmpl<false> (w, mu_e, mu_h, v_1, sys);
}

std::vector<double> plasmon_green_ksq_f(double k2, plasmon_potcoef_s * s) {
    std::complex<double> w_complex(s->w, s->delta);

    double E[2] = { s->sys.m_pe * k2, s->sys.m_ph * k2 };

    std::complex<double> nu[4] = {
        -w_complex / E[0] - 1.0, -w_complex / E[0] + 1.0,
        -w_complex / E[1] - 1.0, -w_complex / E[1] + 1.0
    };

    std::complex<double> pi_screen_nofactor[2] = {
        -2.0 - nu[0] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[0] * nu[0])) + nu[1] * std::sqrt(1.0 - 4.0 * s->mu_e / (E[0] * nu[1] * nu[1])),
        -2.0 - nu[2] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[2] * nu[2])) + nu[3] * std::sqrt(1.0 - 4.0 * s->mu_h / (E[1] * nu[3] * nu[3]))
    };

    std::complex<double> green = 1.0 / (
        - s->sys.eps_r * s->v_1 * std::sqrt(k2)
        + (
         pi_screen_nofactor[0] / s->sys.m_pe
         + pi_screen_nofactor[1] / s->sys.m_ph
        ) * 0.125 * M_1_PI
    );

    return { green.real(), green.imag() };
}

double plasmon_potcoef_fr(double th, void * params) {
    plasmon_potcoef_s * s{ static_cast<plasmon_potcoef_s *>(params) };

    double k2{ s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th) };

    std::vector<double> green = plasmon_green_ksq_f(k2, s);

    return green[0];
}

double plasmon_potcoef_fi(double th, void * params) {
    plasmon_potcoef_s * s{ static_cast<plasmon_potcoef_s *>(params) };

    double k2{ s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th) };

    std::vector<double> green = plasmon_green_ksq_f(k2, s);

    return green[1];
}

std::vector<double> plasmon_potcoef(
    const std::vector<double> & wkk,
    double mu_e,
    double mu_h, double v_1,
    const system_data & sys, double delta
) {
    constexpr uint32_t n_int{ 2 };
    double result[n_int] = { 0 }, error[n_int] = { 0 };

    /*
     * wkk -> (w, k1, k2)
     */

    double k2{ wkk[2] };

    if(std::abs(std::abs(wkk[1] / wkk[2]) - 1.0) < 1e-4) {
        k2 *= 1.0 + 1e-5;
    }

    plasmon_potcoef_s s{ wkk[0], wkk[1], k2, mu_e, mu_h, v_1, sys, delta };

    gsl_function integrands[n_int];

    integrands[0].function = &plasmon_potcoef_fr;
    integrands[0].params = &s;

    integrands[1].function = &plasmon_potcoef_fi;
    integrands[1].params = &s;

    constexpr uint32_t local_ws_size{ (1 << 9) };

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(
        local_ws_size
    );

    gsl_integration_qags(
        integrands, M_PI, 0, global_eps, 0, local_ws_size, ws,
        result, error
    );

    gsl_integration_qags(
        integrands + 1, 0, M_PI, global_eps, 0, local_ws_size,
        ws, result + 1, error + 1
    );

    gsl_integration_workspace_free(ws);

    /*
    for (uint32_t i = 0; i < n_int; i++) {
      printf("%d: %f (%e)\n", i, result[i], error[i]);
    }
    printf("\n");
    */

    return { result[0] / M_PI, result[1] / M_PI };
}

std::complex<double> plasmon_potcoef_cx(const std::vector<double> & wkwk,
    double mu_e, double mu_h, double v_1,
    const system_data & sys, double delta) {
    std::vector<double> elem_vec{ plasmon_potcoef(
        { wkwk[2] - wkwk[0], wkwk[3], wkwk[1] }, mu_e, mu_h, v_1, sys, delta
    ) };

    return { elem_vec[0], elem_vec[1] };
}

double plasmon_potcoef_lwl_f(double th, void * params) {
    plasmon_potcoef_lwl_s * s{ static_cast<plasmon_potcoef_lwl_s *>(params) };

    double k2{ s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th) };

    return - 1.0 / (s->v_1 * s->sys.eps_r * std::sqrt(k2) + s->ls);
}

double plasmon_potcoef_lwl(
    const std::vector<double> & kk,
    double ls, double v_1,
    const system_data & sys
) {
    constexpr uint32_t n_int{ 1 };
    double result[n_int] = { 0 }, error[n_int] = { 0 };

    /*
     * kk -> (k1, k2)
     */

    plasmon_potcoef_lwl_s s{ kk[0], kk[1], ls, v_1, sys };

    gsl_function integrands[n_int];

    integrands[0].function = &plasmon_potcoef_lwl_f;
    integrands[0].params = &s;

    constexpr uint32_t local_ws_size{ (1 << 9) };

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(
        local_ws_size
    );

    gsl_integration_qags(
        integrands, M_PI, 0, global_eps, 0, local_ws_size, ws,
        result, error
    );

    gsl_integration_workspace_free(ws);

    /*
    for (uint32_t i = 0; i < n_int; i++) {
      printf("%d: %f (%e)\n", i, result[i], error[i]);
    }
    printf("\n");
    */

    return result[0] / M_PI;
}

std::complex<double> plasmon_sysmat(
    const plasmon_elem_s & elem, double dw,
    double dk, const std::complex<double> & z
) {
    std::complex<double> G0{ z - std::pow(elem.wkwk[1], 2) };

    std::complex<double> result{
        static_cast<double>(kron(elem.id[0], elem.id[2]) * kron(elem.id[1], elem.id[3]))
        + elem.wkwk[1] * elem.val / G0 * dw * dk
    };

    return result;
}

std::vector<double> plasmon_sysmat_full(
    const std::vector<uint32_t> & ids,
    const std::vector<double> & wkwk,
    double dw, double dk,
    const std::complex<double> & z,
    double mu_e, double mu_h, double v_1,
    const system_data & sys
) {
    std::complex<double> G0{ z - std::pow(wkwk[1], 2) };

    std::vector<double> elem_vec{ plasmon_potcoef(
        { wkwk[2] - wkwk[0], wkwk[3], wkwk[1] }, mu_e, mu_h, v_1, sys) };

    std::complex<double> elem{ elem_vec[0], elem_vec[1] };

    std::complex<double> result{
        static_cast<double>(kron(ids[0], ids[2]) * kron(ids[1], ids[3]))
        + wkwk[1] * elem / G0 * dw * dk
    };

    return { result.real(), result.imag() };
}

std::complex<double> plasmon_sysmat_det(
    const std::vector<std::complex<double>> & elems,
    const std::vector<uint32_t> & shape
) {
    arma::cx_mat mat{ arma::cx_mat(&elems[0], shape[0], shape[1]) };

    return arma::det(mat);
}

std::complex<double> plasmon_sysmat_det_zero_fc(
    const std::complex<double> & z,
    const plasmon_sysmat_det_zero_s & s
) {
    s.mat_z_G0.fill(-z);
    s.mat_z_G0 += s.mat_G0;
    s.mat_potcoef = s.mat_kron + s.mat_elem / s.mat_z_G0;

    return arma::det(s.mat_potcoef);
}

std::vector<std::complex<double>> plasmon_sysmat_det_v(
        const std::vector<std::complex<double>> & z_vec,
        const std::vector<std::vector<double>> & wkwk,
        const std::vector<std::vector<uint32_t>> & ids, double dk,
        double dw, uint32_t N_k, uint32_t N_w, double mu_e,
        double mu_h, double v_1, const system_data & sys,
        double delta
) {
    uint32_t N_total{ N_k * N_w * N_k * N_w };
    uint64_t N_z{ z_vec.size() };

    // Constant
    arma::cx_mat mat_elem(N_k * N_w, N_k * N_w);
    arma::cx_mat mat_kron(N_k * N_w, N_k * N_w);
    arma::cx_mat mat_G0(N_k * N_w, N_k * N_w);

    // Change every step
    arma::cx_mat mat_z_G0(N_k * N_w, N_k * N_w);
    arma::cx_mat mat_potcoef(N_k * N_w, N_k * N_w);

    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    // omp_set_num_threads(32);

    #pragma omp parallel for
    for(uint32_t i = 0; i < N_total; i++) {
        const std::vector<uint32_t> & t_id{ ids[i] };
        const std::vector<double> & t_v{ wkwk[i] };

        mat_elem(t_id[0] + N_w * t_id[1], t_id[2] + N_w * t_id[3]) =
            t_v[1] * plasmon_potcoef_cx(t_v, mu_e, mu_h, v_1, sys, delta) * dw * dk;
        mat_kron(t_id[0] + N_w * t_id[1], t_id[2] + N_w * t_id[3]) =
            kron(t_id[0], t_id[2]) * kron(t_id[1], t_id[3]);

        mat_G0(t_id[0] + N_w * t_id[1], t_id[2] + N_w * t_id[3]) =
            -std::pow(t_v[1], 2);
    }

    struct plasmon_sysmat_det_zero_s s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    std::vector<std::complex<double>> result(N_z);

    //#pragma omp parallel for
    for(uint32_t i = 0; i < N_z; i++) {
        result[i] = plasmon_sysmat_det_zero_fc(z_vec[i], s);
    }

    return result;
}

double plasmon_sysmat_det_zero_f(double z, void * params) {
    plasmon_sysmat_det_zero_s * s{
        static_cast<plasmon_sysmat_det_zero_s *>(params)
    };

    s->mat_z_G0.fill(-z);
    s->mat_z_G0 += s->mat_G0;
    s->mat_potcoef = s->mat_kron + s->mat_elem / s->mat_z_G0;

    return arma::det(s->mat_potcoef).real();
}

std::complex<double> plasmon_sysmat_det_zero(
    const std::vector<std::vector<double>> & wkwk,
    const std::vector<std::vector<uint32_t>> & ids,
    double dk, double dw, uint32_t N_k, uint32_t N_w,
    double mu_e, double mu_h, double v_1,
    const system_data & sys, double delta) {
    uint32_t N_total{ N_k * N_w * N_k * N_w };

    // Constant
    arma::cx_mat mat_elem(N_k * N_w, N_k * N_w);
    arma::cx_mat mat_kron(N_k * N_w, N_k * N_w);
    arma::cx_mat mat_G0(N_k * N_w, N_k * N_w);

    // Change every step
    arma::cx_mat mat_z_G0(N_k * N_w, N_k * N_w);
    arma::cx_mat mat_potcoef(N_k * N_w, N_k * N_w);

    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    // omp_set_num_threads(32);
    #pragma omp parallel for
    for(uint32_t i = 0; i < N_total; i++) {
        const std::vector<uint32_t> & t_id{ ids[i] };
        const std::vector<double> & t_v{ wkwk[i] };

        mat_elem(
            t_id[0] + N_w * t_id[1],
            t_id[2] + N_w * t_id[3]
        ) = t_v[1] * plasmon_potcoef_cx(t_v, mu_e, mu_h, v_1, sys, delta) * dw * dk;

        mat_kron(
            t_id[0] + N_w * t_id[1],
            t_id[2] + N_w * t_id[3]
        ) = kron(t_id[0], t_id[2]) * kron(t_id[1], t_id[3]);

        mat_G0(
            t_id[0] + N_w * t_id[1],
            t_id[2] + N_w * t_id[3]
        ) = -std::pow(t_v[1], 2);
    }

    struct plasmon_sysmat_det_zero_s s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    gsl_function funct;
    funct.function = &plasmon_sysmat_det_zero_f;
    funct.params = &s;

    double z{ 0 };
    double z_min{ 1e2 }, z_max{ 2e2 };
    double f1;

    while(z_max > 1e-14) {
        f1 = funct.function(z_min, funct.params);

        if(f1 < 0) {
            break;
        } else {
            z_max = z_min;
            z_min = 0.5 * z_max;
        }
    }

    if (funct.function(z_max, funct.params) * f1 > 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for(int status = GSL_CONTINUE, iter = 0;
        status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z = gsl_root_fsolver_root(solver);
        z_min = gsl_root_fsolver_x_lower(solver);
        z_max = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return -z;
}

double plasmon_sysmat_det_zero_lwl_f(double z, void * params) {
    plasmon_sysmat_det_zero_lwl_s * s{
        static_cast<plasmon_sysmat_det_zero_lwl_s *>(params)
    };

    s->mat_z_G0.fill(-z);
    s->mat_z_G0 += s->mat_G0;
    s->mat_potcoef = s->mat_kron + s->mat_elem / s->mat_z_G0;

    return arma::det(s->mat_potcoef);
}

std::vector<double> plasmon_sysmat_det_lwl_v(
    const std::vector<double> & z_vec,
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids, double dk,
    uint32_t N_k, double ls, double v_1, const system_data & sys
) {
    uint32_t N_total{ N_k * N_k };
    uint64_t N_z{ z_vec.size() };

    // Constant
    arma::mat mat_elem(N_k, N_k);
    arma::mat mat_kron(N_k, N_k);
    arma::mat mat_G0(N_k, N_k);

    // Change every step
    arma::mat mat_z_G0(N_k, N_k);
    arma::mat mat_potcoef(N_k, N_k);

    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    // omp_set_num_threads(32);
    #pragma omp parallel for
    for(uint32_t i = 0; i < N_total; i++) {
        const std::vector<uint32_t> & t_id{ ids[i] };
        const std::vector<double> & t_v{ kk[i] };

        mat_elem(t_id[0], t_id[1]) =
            t_v[1] * plasmon_potcoef_lwl(t_v, ls, v_1, sys) * dk;
        mat_kron(t_id[0], t_id[1]) = kron(t_id[0], t_id[1]);
        mat_G0(t_id[0], t_id[1]) = -std::pow(t_v[1], 2);
    }

    struct plasmon_sysmat_det_zero_lwl_s s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    std::vector<double> result(N_z);

    //#pragma omp parallel for
    for(uint32_t i = 0; i < N_z; i++) {
        result[i] = plasmon_sysmat_det_zero_lwl_f(z_vec[i], &s);
    }

    return result;
}

std::vector<double> plasmon_sysmat_lwl_m(
    double z,
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids, double dk,
    double ls, double v_1, const system_data & sys
) {
    uint64_t N_total{ kk.size() };
    std::vector<double> result(N_total);

    #pragma omp parallel for
    for(uint64_t i = 0; i < N_total; i++) {
        const std::vector<uint32_t> & t_id{ ids[i] };
        const std::vector<double> & t_v{ kk[i] };

        result[i] = kron(t_id[0], t_id[1]) + t_v[1] * plasmon_potcoef_lwl(t_v, ls, v_1, sys) * dk / (z - std::pow(t_v[1], 2));
    }

    return result;
}

std::vector<std::complex<double>> plasmon_sysmat_lwl_eigvals(
    double z,
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids, double dk,
    uint32_t N_k, double ls, double v_1, const system_data & sys
) {
    uint64_t N_total{ kk.size() };

    // Change every step
    arma::mat mat_potcoef(N_k, N_k);

    // omp_set_num_threads(32);
    #pragma omp parallel for
    for(uint32_t i = 0; i < N_total; i++) {
        const std::vector<uint32_t> & t_id{ ids[i] };
        const std::vector<double> & t_v{ kk[i] };

        mat_potcoef(t_id[0], t_id[1]) = kron(t_id[0], t_id[1]) + t_v[1] * plasmon_potcoef_lwl(t_v, ls, v_1, sys) * dk / (z - std::pow(t_v[1], 2));
    }

    arma::cx_vec eigvals = arma::eig_gen(mat_potcoef);
    std::vector<std::complex<double>> result(eigvals.begin(), eigvals.end());

    return result;
}

double plasmon_sysmat_det_zero_lwl(
    const std::vector<std::vector<double>> & kk,
    const std::vector<std::vector<uint32_t>> & ids,
    double dk, uint32_t N_k, double ls, double v_1,
    const system_data & sys
) {
    uint32_t N_total{ N_k * N_k };

    // Constant
    arma::mat mat_elem(N_k, N_k);
    arma::mat mat_kron(N_k, N_k, arma::fill::eye);
    arma::mat mat_G0(N_k, N_k);

    // Change every step
    arma::mat mat_z_G0(N_k, N_k);
    arma::mat mat_potcoef(N_k, N_k);

    /*
     * mat_z_G0 = z - mat_G0
     * mat_potcoef = mat_kron + mat_elem / mat_z_G0
     */

    // omp_set_num_threads(32);
    #pragma omp parallel for
    for(uint32_t i = 0; i < N_total; i++) {
        const std::vector<uint32_t> & t_id{ ids[i] };
        const std::vector<double> & t_v{ kk[i] };

        mat_elem(t_id[0], t_id[1]) =
            t_v[1] * plasmon_potcoef_lwl(t_v, ls, v_1, sys) * dk;
        mat_G0(t_id[0], t_id[1]) = -std::pow(t_v[1], 2);
    }

    struct plasmon_sysmat_det_zero_lwl_s s {
        mat_elem, mat_kron, mat_G0, mat_z_G0, mat_potcoef
    };

    gsl_function funct;
    funct.function = &plasmon_sysmat_det_zero_lwl_f;
    funct.params = &s;

    /*
    if (funct.function(1e-14, funct.params) > 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    */

    double z{ 0 };
    double z_min{ 1e5 }, z_max{ 2e5 };
    double f1;

    while(z_max > 1e-14) {
        f1 = funct.function(z_min, funct.params);

        if(f1 < 0) {
            break;
        } else {
            z_max = z_min;
            z_min = 0.5 * z_max;
        }
    }

    if (funct.function(z_max, funct.params) * f1 > 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const gsl_root_fsolver_type * T = gsl_root_fsolver_brent;
    gsl_root_fsolver * solver = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(solver, &funct, z_min, z_max);

    for(int status = GSL_CONTINUE, iter = 0;
        status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(solver);
        z = gsl_root_fsolver_root(solver);
        z_min = gsl_root_fsolver_x_lower(solver);
        z_max = gsl_root_fsolver_x_upper(solver);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
    }

    gsl_root_fsolver_free(solver);
    return -z;
}

double plasmon_real_potcoef_k_f(double k, void * params) {
    plasmon_real_potcoef_k_s * s{static_cast<plasmon_real_potcoef_k_s*>(params)};

    std::vector<double> elem{
        plasmon_green(
            0, k / s->x,
            s->mu_e, s->mu_h,
            s->v_1, s->sys,
            0
        )
    };

    return k * elem[0] * gsl_sf_bessel_J0(k);

    //return - k / (k / s->x + 5) * gsl_sf_bessel_J0(k);
}

double plasmon_real_potcoef_k_int(
    double x,
    double mu_e, double mu_h, double v_1,
    const system_data & sys
) {
    constexpr uint32_t n_int{ 1 };
    constexpr uint32_t local_ws_size{ (1 << 9) };

    double result[n_int] = { 1 }, error[n_int] = { 0 };
    const uint32_t n_sum { 1<<7 };

    double result_sum = { 0 };
    double last_zero = { 0 };
    double t[n_sum];

    plasmon_real_potcoef_k_s s{ x, mu_e, mu_h, v_1, sys };

    gsl_function integrands[n_int];

    integrands[0].function = &plasmon_real_potcoef_k_f;
    integrands[0].params = &s;

    gsl_integration_workspace * ws = gsl_integration_workspace_alloc(
        local_ws_size
    );

    uint32_t i{ 0 };
    //for (uint32_t i = 0; i < n_sum && abs(result[0]) > 1e-12; i++) {
    while(abs(result[0]) > 1e-6 && i < n_sum) {
        double temp = gsl_sf_bessel_zero_J0(i + 1);

        gsl_integration_qag(
            integrands, last_zero, temp,
            global_eps, 0, local_ws_size, GSL_INTEG_GAUSS31,
            ws, result, error
        );

        t[i] = result[0];
        last_zero = temp;
        //result_sum += result[0];

        i++;
    }

    gsl_sum_levin_u_workspace * w = gsl_sum_levin_u_alloc(n_sum);
    gsl_sum_levin_u_accel(t, i, w, &result_sum, error);
    gsl_sum_levin_u_free(w);

    return result_sum * 0.5 * M_1_PI / ( x * x );
}

std::vector<double> plasmon_real_potcoef_k(
    const std::vector<double> & x_vec,
    double mu_e, double mu_h, double v_1,
    const system_data & sys
) {
    uint64_t N_x{x_vec.size()};
    std::vector<double> output(N_x);

    #pragma omp parallel for
    for (uint64_t i = 0; i < N_x; i++) {
        output[i] = plasmon_real_potcoef_k_int(
            x_vec[i],
            mu_e, mu_h, v_1,
            sys
        );
    }

    return output;
}
