#pragma once

#include "common.h"

#ifndef SWIG

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

template <uint32_t N, typename F, typename ... Args> std::vector<double> derivative_c2(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F

  std::vector<double> fp_val(N);
  std::vector<double> f_plus{f(x0 + h, args...)};
  std::vector<double> f_minus{f(x0 - h, args...)};

  for (uint32_t j = 0; j < N; j++) {
    fp_val[j] = (f_plus[j] - f_minus[j]) / (2 * h);
  }

  return fp_val;
}

template <uint32_t N, typename F, typename ... Args> std::vector<double> derivative_f3(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F

  constexpr uint32_t n_ord = 3;
  constexpr double pf[n_ord] = {-1.5, 2, -0.5};

  std::vector<double> fp_val(N);
  std::vector<double> f_vals(N, 0);

  for (uint32_t i = 0; i < n_ord; i++) {
    f_vals = f(x0 + i * h, args...);

    for (uint32_t j = 0; j < N; j++) {
      fp_val[j] += pf[i] * f_vals[j];
    }
  }

  for (uint32_t i = 0; i < N; i++) {
    fp_val[i] /= h;
  }

  return fp_val;
}

template <uint32_t N, typename F, typename ... Args> std::vector<double> derivative_c5(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F
  constexpr double p = 1;

  constexpr double pf[] = {
      (2 * p*p*p - 3*p*p -   p + 1) / 12.0,
    - (4 * p*p*p - 3*p*p - 8*p + 4) / 6.0,
      (2 * p*p*p         - 5*p    ) / 2.0,
    - (4 * p*p*p + 3*p*p - 8*p - 4) / 6.0,
      (2 * p*p*p + 3*p*p -   p - 1) / 12.0
  };

  std::vector<double> fp_val(N);
  std::vector<double> temp(N);

  for (uint32_t j = 0; j < N; j++) {
    fp_val[j] = 0;

    for (int32_t i = 0; i < 5; i++) {
      temp = f(x0 + (i - 2)*h, args...);

      fp_val[j] += pf[i] * temp[j];
    }

    fp_val[j] /= h;
  }

  return fp_val;
}

#endif

