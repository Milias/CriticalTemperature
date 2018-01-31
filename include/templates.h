#pragma once

#include "common.h"

template <uint32_t local_ws_size = w_size, typename F, typename ... Args> double integralTemplate(const F & integrand_function, double x0, Args ... args) {
  double result[] = {0}, error;
  double params_arr[sizeof...(args)] = { args... };

  gsl_function integrand;
  integrand.function = integrand_function;
  integrand.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(local_ws_size);
  gsl_integration_qagiu(&integrand, x0, 0, 1e-10, local_ws_size, ws, result, &error);
  gsl_integration_workspace_free(ws);

  return result[0];
}

template <uint32_t N, typename F, typename ... Args> double * derivative_c2(const F & f, double x0, double h, Args ... args) {
  // double * is return type of F

  double * fp_val = new double[N];
  double * f_plus = f(x0 + h, args...);
  double * f_minus = f(x0 - h, args...);

  for (uint32_t j = 0; j < N; j++) {
    fp_val[j] = (f_plus[j] - f_minus[j]) / (2 * h);
  }

  delete[] f_plus;
  delete[] f_minus;

  return fp_val;
}

template <uint32_t N, typename F, typename ... Args> double * derivative_f3(const F & f, double x0, double h, Args ... args) {
  // double * is return type of F

  constexpr uint32_t n_ord = 3;
  constexpr double pf[n_ord] = {-1.5, 2, -0.5};

  double * fp_val = new double[N];
  double * f_vals;

  for (uint32_t i = 0; i < N; i++) {
    fp_val[i] = 0;
  }

  for (uint32_t i = 0; i < n_ord; i++) {
    f_vals = f(x0 + i * h, args...);

    for (uint32_t j = 0; j < N; j++) {
      fp_val[j] += pf[i] * f_vals[j];
    }

    delete[] f_vals;
  }

  for (uint32_t i = 0; i < N; i++) {
    fp_val[i] /= h;
  }

  return fp_val;
}

template <uint32_t N, typename F, typename ... Args> double * derivative_c5(const F & f, double x0, double h, Args ... args) {
  // double * is return type of F

  constexpr double p = 1;

  constexpr double pf[] = {
      (2 * p*p*p - 3*p*p -   p + 1) / 12.0,
    - (4 * p*p*p - 3*p*p - 8*p + 4) / 6.0,
      (2 * p*p*p         - 5*p    ) / 2.0,
    - (4 * p*p*p + 3*p*p - 8*p - 4) / 6.0,
      (2 * p*p*p + 3*p*p -   p - 1) / 12.0
  };

  double * fp_val = new double[N];

  for (uint32_t j = 0; j < N; j++) {
    fp_val[j] = 0;

    for (int32_t i = 0; i < 5; i++) {
      double * temp = f(x0 + (i - 2)*h, args...);

      fp_val[j] += pf[i] * temp[j];

      delete[] temp;
    }

    fp_val[j] /= h;
  }

  return fp_val;
}
