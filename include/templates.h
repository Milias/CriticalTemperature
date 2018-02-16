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

template <uint32_t N = 1, typename F, typename ... Args> auto derivative_c2(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F

  if constexpr(N > 1) {
    std::vector<double> fp_val(N);

    std::vector<double> f_plus{f(x0 + h, args...)};
    std::vector<double> f_minus{f(x0 - h, args...)};

    for (uint32_t j = 0; j < N; j++) {
      fp_val[j] = (f_plus[j] - f_minus[j]) / (2 * h);
    }

    return fp_val;
  } else {
    return (f(x0 + h, args...) - f(x0 - h, args...)) / (2 * h);
  }
}

template <uint32_t n_ord, uint32_t N = 1, typename F, typename ... Args> auto derivative_generic(double * pf, int32_t * ip, const F & f, double x0, double h, Args ... args) {
  if constexpr(N > 1) {
    std::vector<double> fp_val(N);
    std::vector<double> f_vals(N, 0);

    for (uint32_t i = 0; i < n_ord; i++) {
      f_vals = f(x0 + ip[i] * h, args...);

      for (uint32_t j = 0; j < N; j++) {
        fp_val[j] += pf[i] * f_vals[j];
      }
    }

    for (uint32_t i = 0; i < N; i++) {
      fp_val[i] /= h;
    }

    return fp_val;
  } else {
    double f_vals{0};

    for (uint32_t i = 0; i < n_ord; i++) {
      f_vals += pf[i] * f(x0 + ip[i] * h, args...);
    }

    return f_vals / h;
  }
}

template <uint32_t N = 1, typename F, typename ... Args> auto derivative_c4(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F

  constexpr uint32_t n_ord = 4;
  constexpr double pf[n_ord] = {1.0/12, -8.0/12, 8.0/12, -1.0/12};
  constexpr int32_t ip[n_ord] = {-2, -1, 1, 2};

  return derivative_generic<n_ord, N>(pf, ip, f, x0, h, std::forward<Args...>(args...));
}

template <uint32_t N = 1, typename F, typename ... Args> auto derivative_f3(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F

  constexpr uint32_t n_ord = 3;
  constexpr double pf[n_ord] = {-1.5, 2, -0.5};
  constexpr int32_t ip[n_ord] = {0, 1, 2};

  return derivative_generic<n_ord, N>(pf, ip, f, x0, h, std::forward<Args...>(args...));
}

template <uint32_t N = 1, typename F, typename ... Args> auto derivative_b3(const F & f, double x0, double h, Args ... args) {
  // std::vector<double> is return type of F

  constexpr uint32_t n_ord = 3;
  constexpr double pf[n_ord] = {1.5, -2.0, 0.5};
  constexpr int32_t ip[n_ord] = {0, -1, -2};

  return derivative_generic<n_ord, N>(pf, ip, f, x0, h, std::forward<Args...>(args...));
}

template <typename T, typename F, typename Arg1, typename ... Args> inline T python_wrap(const F & f, Arg1 arg1, Args ... args) {
  double params_arr[sizeof...(args)] = { args... };
  return f(arg1, params_arr);
}

#endif

