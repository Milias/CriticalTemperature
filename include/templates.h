#pragma once

#include "common.h"

template <typename F, typename ... Args> double integralTemplate(const F & integrand_function, Args ... args) {
  double result[] = {0}, error;
  double params_arr[sizeof...(args)] = { args... };

  gsl_function integrand;
  integrand.function = integrand_function;
  integrand.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
  gsl_integration_qagiu(&integrand, 0, 0, 1e-10, w_size, ws, result, &error);
  gsl_integration_workspace_free(ws);

  return result[0];
}

template <typename F, typename ... Args> double integralTemplate(const F & integrand_function, double x0, Args ... args) {
  double result[] = {0}, error;
  double params_arr[sizeof...(args)] = { args... };

  gsl_function integrand;
  integrand.function = integrand_function;
  integrand.params = params_arr;

  gsl_integration_workspace * ws = gsl_integration_workspace_alloc(w_size);
  gsl_integration_qagiu(&integrand, x0, 0, 1e-10, w_size, ws, result, &error);
  gsl_integration_workspace_free(ws);

  return result[0];
}

