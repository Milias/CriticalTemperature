#pragma once

#include <acb.h>
#include <acb_hypgeom.h>
#include <arb.h>
#include <arb_hypgeom.h>
#include <arf.h>
#include <assert.h>
#include <gmp.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_dawson.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_zeta.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_sum.h>
#include <gsl/gsl_vector.h>
#include <mpfr.h>
#include <omp.h>

#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <ratio>
#include <thread>
#include <typeinfo>
#include <utility>
#include <string>
#include <filesystem>

//#include <optim/optim.hpp>
#include <armadillo>

#include "Faddeeva.hh"
#include "utils.h"

/*** MPFR ***/

constexpr bool use_mpfr{false};
constexpr mp_prec_t prec{64};

/*** gsl constants ***/

constexpr double global_eps{1e-10};
constexpr int max_iter{32};

/*** Initialization ***/

void initializeMPFR_GSL();

/*** Utility functions ***/

double logExp(double x);
double logExp_mpfr(double x, double xmax = 1e3);

// real(Li_s(exp(z)))
double polylogExp(double s, double z);

// real(Li_s(-exp(z)))
double polylogExpM(double s, double z);

// find z such that PolyLog[s, Exp[z]] == a.
double invPolylogExp(double s, double a);

// find z such that -PolyLog[s, -Exp[z]] == a.
double invPolylogExpM(double s, double a);

// https://math.stackexchange.com/questions/712434/erfaib-error-function-separate-into-real-and-imaginary-part
double erf_fk(double x, double y, uint32_t k);
double erf_gk(double x, double y, uint32_t k);

double erf_sterm_r(double x, double y, double k);
double erf_sterm_i(double x, double y, double k);

// real(erf(x + i * y))
double erf_r(double x, double y, double eps = 1e-16);

// imag(erf(x + i * y))
double erf_i(double x, double y, double eps = 1e-16);

std::complex<double> erf_cx(const std::complex<double>& z);
std::complex<double> erfi_cx(const std::complex<double>& z);

/*
 * Solve the equation
 *
 * y^2 / (1 - y) == v
 */
double y_eq_s(double v);

/*
 * Return type for integrals.
 */
template <size_t N_INT>
struct result_s {
    double value[N_INT] = {0}, error[N_INT] = {0};
    size_t neval[N_INT] = {0};

    uint32_t n_int{N_INT};

    result_s<N_INT>() {}
    result_s<N_INT>(double val) { value[0] = val; }

    static result_s<N_INT> nan() {
        result_s<N_INT> r;
        for (size_t i = 0; i < N_INT; i++) {
            r.value[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return r;
    }

    template <size_t N_NEXT>
    double add_next(const double* a_ptr) const {
        if constexpr (N_NEXT > 1) {
            return a_ptr[N_NEXT - 1] + add_next<N_NEXT - 1>(a_ptr);
        } else {
            return a_ptr[N_NEXT - 1];
        }
    }

    double total_value() const { return add_next<N_INT>(value); }

    double total_error() const { return add_next<N_INT>(error); }
};

#ifndef SWIG

template <uint32_t N, typename T>
T sum_result(T* values) {
    if constexpr (N > 1) {
        return values[N - 1] + sum_result<N - 1, T>(values);
    } else {
        return values[0];
    }
}

template <typename T>
double functor_call(double x, void* params) {
    T* s{static_cast<T*>(params)};

    return (*s)(x);
}

template <typename T, double (*F)(double, T*)>
double templated_f(double int_var, void* params) {
    T* s{static_cast<T*>(params)};

    return F(int_var, s);
}

template <typename T, double (*F)(double, T*), uint32_t M_1_H = 1000000>
double templated_df(double int_var, void* params) {
    // M_1_H = 1 / h
    constexpr double h{1.0 / M_1_H};

    T* s{static_cast<T*>(params)};

    return 0.5 * M_1_H * (F(int_var + h, s) - F(int_var - h, s));
}

template <typename T, void (*F)(double, T*, double*, double*)>
void templated_fdf(double int_var, void* params, double* f, double* df) {
    T* s{static_cast<T*>(params)};
    F(int_var, s, f, df);
}

template <typename T, double (*F)(double, T*), uint32_t M_1_H = 1000000>
void templated_fdf(double int_var, void* params, double* f, double* df) {
    T* s{static_cast<T*>(params)};

    *f  = F(int_var, s);
    *df = templated_df<T, F, M_1_H>(int_var, params);
}

template <typename T, double (*F)(double, T*), double (*dF)(double, T*)>
void templated_fdf(double int_var, void* params, double* f, double* df) {
    T* s{static_cast<T*>(params)};

    *f  = F(int_var, s);
    *df = dF(int_var, s);
}

/*
 * Only for use in conjunction with gsl_function.
 */
template <
    typename T = void,
    double (*F)(double, T*),
    uint32_t N_med       = 2,
    uint32_t d_med_denom = 10000>
double median_f(double x, T* s) {
    std::vector<double> result(N_med);

    for (uint32_t i = 0; i < N_med; i++) {
        result[i] =
            F(x + (i - N_med * 0.5) / static_cast<double>(d_med_denom), s);
    }

    std::sort(result.begin(), result.end());

    double med;

    constexpr uint32_t mid_point{(N_med + 1) / 2};

    if constexpr (N_med % 2 == 0) {
        med = 0.5 * (result[mid_point] + result[mid_point + 1]);
    } else {
        med = result[mid_point];
    }

    return med;
}

// real(erf(x + i * y))
template <uint32_t n = 32>
double erf_r_t(double x, double y, double eps = 1e-16);

// imag(erf(x + i * y))
template <uint32_t n = 32>
double erf_i_t(double x, double y, double eps = 1e-16);

template <uint32_t n = 32>
std::complex<double> erf_cx_t(const std::complex<double>& z);

/*
 * Struct used when saving the full solution
 * for an ODE.
 */

template <typename state, typename time = double>
struct ode_observer {
    std::vector<state>& f_vec;
    std::vector<time>& t_vec;

    void operator()(const state& f, time t) {
        f_vec.push_back(f);
        t_vec.push_back(t);
    }
};

template <typename T>
T constexpr sqrtNewtonRaphson(T x, T curr, T prev) {
    return curr == prev
               ? curr
               : sqrtNewtonRaphson<T>(x, 0.5 * (curr + x / curr), curr);
}

/*
 * Constexpr version of the square root
 * Return value:
 *   - For a finite and non-negative value of "x", returns an approximation for
 * the square root of "x"
 *   - Otherwise, returns NaN
 *
 * https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
 */
template <typename T>
T constexpr const_sqrt(T x) {
    return x >= 0 && x < std::numeric_limits<T>::infinity()
               ? sqrtNewtonRaphson<T>(x, x, 0)
               : std::numeric_limits<T>::quiet_NaN();
}

std::complex<double> erf_c(std::complex<double>& z);

template <typename T>
T constexpr sign(T x) {
    return x == 0 ? 0 : (x > 0 ? 1 : -1);
}
template <typename T>
T constexpr kron(T a, T b) {
    return a == b ? 1 : 0;
}

#endif

/*** Struve Function ***/

double struve(double v, double x);

