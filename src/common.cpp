#include "common.h"
#include "common_utils.h"

/*** Initialization ***/

void initializeMPFR_GSL() {
    mpfr_set_default_prec(prec);
    gsl_set_error_handler_off();
}

/*** Utility functions ***/

double logExp(double x, double xmax) {
    // log( 1 + exp( x ) )
    double d_y;

    if (x < xmax) {
        // Approximate log(exp(x) + 1) ~ x when x > xmax
        d_y = log(1 + exp(x));
    } else {
        d_y = x;
    }

    return d_y;
}

double logExp_mpfr(double x, double xmax) {
    // log( 1 + exp( x ) )
    double d_y;

    if (x < xmax) {
        // Approximate log(exp(x) + 1) ~ x when x > xmax
        mpfr_t y;
        mpfr_init_set_d(y, x, MPFR_RNDN);
        mpfr_exp(y, y, MPFR_RNDN);
        mpfr_add_ui(y, y, 1, MPFR_RNDN);
        mpfr_log(y, y, MPFR_RNDN);

        d_y = mpfr_get_d(y, MPFR_RNDN);

        mpfr_clear(y);
    } else {
        d_y = x;
    }

    return d_y;
}

double polylogExp(double s, double z) {
    double r{0};
    // assert(z <= 0 && "z has to be negative.");

    if (z >= 0) {
        arb_t arb_x, arb_s, arb_z;

        arb_init(arb_x);
        arb_init(arb_s);
        arb_init(arb_z);

        arb_set_d(arb_s, s);
        arb_set_d(arb_z, std::exp(z));
        // arb_exp(arb_z, arb_z, prec);

        arb_polylog(arb_x, arb_s, arb_z, prec);

        r = arf_get_d(arb_midref(arb_x), ARF_RND_NEAR);

        arb_clear(arb_x);
        arb_clear(arb_s);
        arb_clear(arb_z);
    } else {
        acb_t acb_x, acb_s, acb_z;

        acb_init(acb_x);
        acb_init(acb_s);
        acb_init(acb_z);

        acb_set_d(acb_s, s);
        acb_set_d(acb_z, std::exp(z));
        // acb_exp(acb_z, acb_z, prec);

        acb_polylog(acb_x, acb_s, acb_z, prec);

        r = arf_get_d(arb_midref(acb_realref(acb_x)), ARF_RND_NEAR);

        acb_clear(acb_x);
        acb_clear(acb_s);
        acb_clear(acb_z);
    }

    return r;
}

double polylogExpM(double s, double z) {
    arb_t arb_x, arb_s, arb_z;

    arb_init(arb_x);
    arb_init(arb_s);
    arb_init(arb_z);

    arb_set_d(arb_s, s);
    arb_set_d(arb_z, -std::exp(z));
    // arb_exp(arb_z, arb_z, prec);
    // arb_neg(arb_z, arb_z);

    arb_polylog(arb_x, arb_s, arb_z, prec);

    double r = -arf_get_d(arb_midref(arb_x), ARF_RND_NEAR);

    arb_clear(arb_x);
    arb_clear(arb_s);
    arb_clear(arb_z);

    return r;
}

double invPolylogExp_f(double z, void* params) {
    double s = ((double*)params)[0];
    double a = ((double*)params)[1];
    return polylogExp(s, z) - a;
}

double invPolylogExp_df(double z, void* params) {
    return polylogExp(((double*)params)[0] - 1.0, z);
}

void invPolylogExp_fdf(double z, void* params, double* f, double* df) {
    double s = ((double*)params)[0];
    double a = ((double*)params)[1];
    f[0]     = polylogExp(s, z) - a;
    df[0]    = polylogExp(s - 1.0, z);
}

double invPolylogExp(double p_s, double a) {
    // Can't be higher than zeta(p_s) == Li(p_s, exp(0))
    if (a < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (a == 0) {
        return -std::numeric_limits<double>::infinity();
    }

    if (p_s > 1) {
        double zeta_val = gsl_sf_zeta(p_s);
        if (a > zeta_val) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (a == zeta_val) {
            return 0;
        }
    }

    double params_arr[] = {p_s, a};
    // TODO: approximate answer for very large or very small a.
    double x0, x = -1e-5;

    gsl_function_fdf T_mat;
    T_mat.f      = &invPolylogExp_f;
    T_mat.df     = &invPolylogExp_df;
    T_mat.fdf    = &invPolylogExp_fdf;
    T_mat.params = params_arr;

    const gsl_root_fdfsolver_type* T = gsl_root_fdfsolver_steffenson;
    gsl_root_fdfsolver* s            = gsl_root_fdfsolver_alloc(T);

    gsl_root_fdfsolver_set(s, &T_mat, x);

    for (int status = GSL_CONTINUE; status == GSL_CONTINUE;) {
        status = gsl_root_fdfsolver_iterate(s);
        x0     = x;
        x      = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, 0, global_eps);
    }

    gsl_root_fdfsolver_free(s);
    return x;
}

double invPolylogExpM_f(double z, void* params) {
    double s = ((double*)params)[0];
    double a = ((double*)params)[1];
    return polylogExpM(s, z) - a;
}

double invPolylogExpM_df(double z, void* params) {
    return polylogExpM(((double*)params)[0] - 1.0, z);
}

void invPolylogExpM_fdf(double z, void* params, double* f, double* df) {
    double s = ((double*)params)[0];
    double a = ((double*)params)[1];
    f[0]     = polylogExpM(s, z) - a;
    df[0]    = polylogExpM(s - 1.0, z);
}

double invPolylogExpM(double p_s, double a) {
    // Can't be negative.
    if (a < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (a == 0) {
        return -std::numeric_limits<double>::infinity();
    }
    if (a > 300) {
        // https://en.wikipedia.org/wiki/Polylogarithm#Limiting_behavior
        return std::pow(a * gsl_sf_gamma(p_s + 1.0), 1 / p_s);
    }

    double params_arr[] = {p_s, a};
    double x0, x = 0;

    gsl_function_fdf func;
    func.f      = &invPolylogExpM_f;
    func.df     = &invPolylogExpM_df;
    func.fdf    = &invPolylogExpM_fdf;
    func.params = params_arr;

    const gsl_root_fdfsolver_type* T = gsl_root_fdfsolver_steffenson;
    gsl_root_fdfsolver* s            = gsl_root_fdfsolver_alloc(T);

    gsl_root_fdfsolver_set(s, &func, x);

    for (int status = GSL_CONTINUE; status == GSL_CONTINUE;) {
        status = gsl_root_fdfsolver_iterate(s);
        x0     = x;
        x      = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, 0, global_eps);
    }

    // printf("%.10f, %.10f, %.3e\n", a, x, x - x0);

    gsl_root_fdfsolver_free(s);
    return x;
}

double erf_fk(double x, double y, uint32_t k) {
    return 2 * x *
           (1 - std::cos(2 * x * y) * std::cosh(k * y) +
            k * std::sin(2 * x * y) * std::sinh(k * y));
}

double erf_gk(double x, double y, uint32_t k) {
    return 2 * x * std::sin(2 * x * y) * std::cosh(k * y) +
           k * std::cos(2 * x * y) * std::sinh(k * y);
}

double erf_sterm_r(double x, double y, double k) {
    return std::exp(-0.25 * k * k) * erf_fk(x, y, k) / (k * k + 4 * x * x);
}

double erf_sterm_i(double x, double y, double k) {
    return std::exp(-0.25 * k * k) * erf_gk(x, y, k) / (k * k + 4 * x * x);
}

template <uint32_t n = 64>
double erf_r(double x, double y, double eps) {
    const double constant_add = std::erf(x) + std::exp(-x * x) /
                                                  (2 * M_PI * x) *
                                                  (1 - std::cos(2 * x * y));
    const double constant_prod = 2 * std::exp(-x * x) / M_PI;
    double val_prev = 0, val_curr, err = 1;

    val_curr = constant_prod * erf_sterm_r(x, y, 1);

    for (uint32_t k = 2; k < n && err > eps; k++) {
        val_prev += val_curr;
        val_curr = constant_prod * erf_sterm_r(x, y, k);

        err = std::abs(val_curr - val_prev);
    }

    return constant_add + val_curr + val_prev;
}

template <uint32_t n = 64>
double erf_i(double x, double y, double eps) {
    const double constant_add =
        std::exp(-x * x) / (2 * M_PI * x) * std::sin(2 * x * y);
    const double constant_prod = 2 * std::exp(-x * x) / M_PI;
    double val_prev = 0, val_curr, err = 1;

    val_curr = constant_prod * erf_sterm_i(x, y, 1);

    for (uint32_t k = 2; k < n && err > eps; k++) {
        val_prev += val_curr;
        val_curr = constant_prod * erf_sterm_i(x, y, k);

        err = std::abs(val_curr - val_prev);
    }

    return constant_add + val_curr + val_prev;
}

template <uint32_t n = 64>
std::complex<double> erf_cx(const std::complex<double>& z, double eps) {
    const std::complex<double> c_add(
        std::erf(z.real()) + std::exp(-std::pow(z.real(), 2)) /
                                 (2 * M_PI * z.real()) *
                                 (1 - std::cos(2 * z.real() * z.imag())),
        std::exp(-std::pow(z.real(), 2)) / (2 * M_PI * z.real()) *
            std::sin(2 * z.real() * z.imag()));

    const double c_prod{2 * std::exp(-std::pow(z.real(), 2)) / M_PI};

    std::complex<double> val_prev{0}, val_curr;
    double err{1};

    val_curr = c_prod * std::complex<double>(
                            erf_sterm_r(z.real(), z.imag(), 1),
                            erf_sterm_i(z.real(), z.imag(), 1));

    for (uint32_t k = 2; k < n && err > eps; k++) {
        val_prev += val_curr;
        val_curr = c_prod * std::complex<double>(
                                erf_sterm_r(z.real(), z.imag(), k),
                                erf_sterm_i(z.real(), z.imag(), k));

        err = std::abs(val_curr - val_prev);
    }

    return c_add + val_curr + val_prev;
}

double y_eq_s_f(double y, void* params) {
    y_eq_s_s* s{static_cast<y_eq_s_s*>(params)};

    return y * y / (1 - y) - s->v;
}

double y_eq_s(double v) {
    // defined in common_utils.h
    y_eq_s_s params{v};
    double z_min{0}, z_max{1 - 1e-12}, z;

    gsl_function funct;
    funct.function = &y_eq_s_f;
    funct.params   = &params;

    const gsl_root_fsolver_type* T = gsl_root_fsolver_brent;
    gsl_root_fsolver* s            = gsl_root_fsolver_alloc(T);

    gsl_root_fsolver_set(s, &funct, z_min, z_max);

    for (int status = GSL_CONTINUE, iter = 0;
         status == GSL_CONTINUE && iter < max_iter; iter++) {
        status = gsl_root_fsolver_iterate(s);
        z      = gsl_root_fsolver_root(s);
        z_min  = gsl_root_fsolver_x_lower(s);
        z_max  = gsl_root_fsolver_x_upper(s);

        status = gsl_root_test_interval(z_min, z_max, 0, global_eps);
        // printf("%f, %f, %f\n", lambda_s, z, funct.function(z, &params));
    }

    gsl_root_fsolver_free(s);
    return z;
}

std::complex<double> erf_c(std::complex<double>& z) {
    return std::complex<double>(
        erf_r(z.real(), z.imag()), erf_i(z.real(), z.imag()));
}
