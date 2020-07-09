#include "topo.h"
#include "topo_utils.h"

arma::vec topo_disp_p(const arma::vec& k_vec, const system_data_v2& sys) {
    return 0.5 * std::pow(sys.c_hbarc, 2) * arma::pow(k_vec, 2) /
           sys.d_params.m_p;
}

arma::vec topo_disp_t(const arma::vec& k_vec, const system_data_v2& sys) {
    arma::vec k2_vec{arma::pow(k_vec, 2)};

    return arma::sqrt(
               4 * std::pow(sys.params.A2, 2) * k2_vec +
               arma::pow(sys.params.B2 * k2_vec - 4 * sys.params.M, 2)) -
           2 * sys.params.A2 *
               std::sqrt(
                   -std::pow(sys.params.A2, 2) +
                   4 * sys.params.B2 * sys.params.M) /
               sys.params.B2;
}

arma::vec topo_disp_t2(const arma::vec& k_vec, const system_data_v2& sys) {
    arma::vec k2_vec{arma::pow(k_vec, 2)};

    return (
        -4 * sys.params.C - k2_vec * sys.params.D2 +
        0.5 * arma::sqrt(
                  4 * std::pow(sys.params.A2, 2) * k2_vec +
                  arma::pow(sys.params.B2 * k2_vec - 4 * sys.params.M, 2)) -
        (-4 * sys.params.B2 *
             (sys.params.B2 * sys.params.C + sys.params.D2 * sys.params.M) +
         sys.params.A2 * (2 * sys.params.A2 * sys.params.D2 +
                          std::sqrt(
                              (4 * std::pow(sys.params.D2, 2) -
                               std::pow(sys.params.B2, 2)) *
                              (std::pow(sys.params.A2, 2) -
                               4 * sys.params.B2 * sys.params.M)))) /
            std::pow(sys.params.B2, 2));
}

double topo_green_cou(double k, const system_data_v2& sys) {
    return -sys.c_hbarc * sys.c_aEM / (sys.params.eps_sol * k);
}

template <double (*green_func)(double, const system_data_v2&)>
double topo_potential_f(double th, topo_potential_s* s) {
    double k{
        std::sqrt(
            s->k1 * s->k1 + s->k2 * s->k2 - 2 * s->k1 * s->k2 * std::cos(th)),
    };

    if (k < 1e-5) {
        k = 1e-5;
    }

    return green_func(k, s->sys);
}

template <double (*green_func)(double, const system_data_v2&)>
double topo_potential_t(const double kk[2], const system_data_v2& sys) {
    constexpr uint32_t n_int{1};
    double result[n_int] = {0}, error[n_int] = {0};
    double k2{kk[1]};

    topo_potential_s s{kk[0], k2, sys};

    gsl_function integrands[n_int];

    integrands[0].function =
        &templated_f<topo_potential_s, topo_potential_f<green_func>>;
    integrands[0].params = &s;

    constexpr uint32_t local_ws_size{(1 << 7)};
    constexpr double local_eps{1e-8};

    gsl_integration_workspace* ws =
        gsl_integration_workspace_alloc(local_ws_size);

    gsl_integration_qag(
        integrands, M_PI, 0, local_eps, 0, local_ws_size, GSL_INTEG_GAUSS31,
        ws, result, error);

    gsl_integration_workspace_free(ws);

    return result[0] * M_1_PI;
}

template <
    double (*potential_func)(const double[2], const system_data_v2&),
    arma::vec (*dispersion_func)(const arma::vec&, const system_data_v2&),
    bool force_positive = true>
double topo_det_f(double z, topo_mat_s<potential_func, dispersion_func>* s) {
    if constexpr (force_positive) {
        z = std::exp(z);
    }

    s->update_mat_potcoef(z);
    return arma::det(s->mat_potcoef);
}

template <
    double (*potential_func)(const double[2], const system_data_v2&),
    arma::vec (*dispersion_func)(const arma::vec&, const system_data_v2&),
    bool use_brackets = false>
double topo_det_zero_t(
    topo_mat_s<potential_func, dispersion_func>& s, double be_bnd) {
    /*
     * be_bnd: upper bound for the binding energy -> positive value!
     */

    constexpr double local_eps{1e-8}, be_min{50e-3};
    double z, z_min{be_min}, z_max{std::abs(be_bnd)}, z0{be_bnd};

    s.fill_mat_potcoef();
    s.fill_mat_elem();

    if constexpr (use_brackets) {
        gsl_function funct;
        funct.function = &median_f<
            void,
            templated_f<
                topo_mat_s<potential_func, dispersion_func>, topo_det_f>,
            7>;
        funct.params = &s;

        double f_min{funct.function(z_min, funct.params)};
        double f_max{funct.function(z_max, funct.params)};

        ///*
        printf(
            "[%s] f: (%e, %e), z: (%f, %f)\n", __func__, f_min, f_max, z_min,
            z_max);
        //*/

        if (f_min * f_max > 0) {
            printf(
                "\033[91;1m[%s] Ends of the bracket must have opposite "
                "signs.\033[0m\n",
                __func__);
            return std::numeric_limits<double>::quiet_NaN();
        }

        const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_brent;
        gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

        gsl_root_fsolver_set(solver, &funct, z_min, z_max);

        for (int status = GSL_CONTINUE, iter = 0;
             status == GSL_CONTINUE && iter < max_iter; iter++) {
            status = gsl_root_fsolver_iterate(solver);
            z      = gsl_root_fsolver_root(solver);
            z_min  = gsl_root_fsolver_x_lower(solver);
            z_max  = gsl_root_fsolver_x_upper(solver);

            ///*
            printf(
                "[%s]: <%d> f: (%e, %e), z: (%f, %f)\n", __func__, iter,
                funct.function(z_min, funct.params),
                funct.function(z_max, funct.params), z_min, z_max);
            //*/

            status = gsl_root_test_interval(z_min, z_max, 0, local_eps);
        }

        gsl_root_fsolver_free(solver);
    } else {
        gsl_function_fdf funct;
        funct.f = &templated_f<
            topo_mat_s<potential_func, dispersion_func>, topo_det_f>;
        funct.df = &templated_df<
            topo_mat_s<potential_func, dispersion_func>, topo_det_f>;
        funct.fdf = &templated_fdf<
            topo_mat_s<potential_func, dispersion_func>, topo_det_f>;
        funct.params = &s;

        if (std::abs(funct.f(z0, funct.params)) < local_eps) {
            return std::exp(z0);
        }

        const gsl_root_fdfsolver_type* solver_type =
            gsl_root_fdfsolver_steffenson;
        gsl_root_fdfsolver* solver = gsl_root_fdfsolver_alloc(solver_type);

        gsl_root_fdfsolver_set(solver, &funct, z0);

        for (int status = GSL_CONTINUE, iter = 0;
             status == GSL_CONTINUE && iter < max_iter; iter++) {
            z0     = z;
            status = gsl_root_fdfsolver_iterate(solver);
            z      = gsl_root_fdfsolver_root(solver);
            status =
                gsl_root_test_residual(funct.f(z, funct.params), local_eps);

            printf(
                "[%s]: <%d> f: (%e), z: (%f)\n", __func__, iter,
                funct.f(z, funct.params), z);
        }

        gsl_root_fdfsolver_free(solver);
        z = std::exp(z);
    }

    return z;
}

std::vector<double> topo_det_p_cou_vec(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys) {
    constexpr auto green_func      = topo_green_cou;
    constexpr auto dispersion_func = topo_disp_p;
    constexpr auto potential_func  = topo_potential_t<green_func>;

    constexpr auto det_func =
        topo_det_f<potential_func, dispersion_func, false>;

    topo_mat_s<potential_func, dispersion_func> mat_s(N_k, sys);

    mat_s.fill_mat_potcoef();
    mat_s.fill_mat_elem();

    std::vector<double> r(z_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < z_vec.size(); i++) {
        r[i] = det_func(z_vec[i], &mat_s);
    }

    return r;
}

std::vector<double> topo_det_t_cou_vec(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys) {
    constexpr auto green_func      = topo_green_cou;
    constexpr auto dispersion_func = topo_disp_t;
    constexpr auto potential_func  = topo_potential_t<green_func>;

    constexpr auto det_func =
        topo_det_f<potential_func, dispersion_func, false>;

    topo_mat_s<potential_func, dispersion_func> mat_s(N_k, sys);

    mat_s.fill_mat_potcoef();
    mat_s.fill_mat_elem();

    std::vector<double> r(z_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < z_vec.size(); i++) {
        r[i] = det_func(z_vec[i], &mat_s);
    }

    return r;
}

double topo_be_p_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd) {
    constexpr auto green_func      = topo_green_cou;
    constexpr auto dispersion_func = topo_disp_p;
    constexpr auto potential_func  = topo_potential_t<green_func>;

    constexpr auto det_zero = topo_det_zero_t<potential_func, dispersion_func>;

    topo_mat_s<potential_func, dispersion_func> mat_s(N_k, sys);

    return det_zero(mat_s, be_bnd);
}

double topo_be_t_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd) {
    constexpr auto green_func      = topo_green_cou;
    constexpr auto dispersion_func = topo_disp_t;
    constexpr auto potential_func  = topo_potential_t<green_func>;

    constexpr auto det_zero = topo_det_zero_t<potential_func, dispersion_func>;

    topo_mat_s<potential_func, dispersion_func> mat_s(N_k, sys);

    return det_zero(mat_s, be_bnd);
}
