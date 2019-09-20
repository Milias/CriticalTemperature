#pragma once

#include "common.h"

template <typename state, double (*pot)(double, const system_data&)>
struct wf_dy_s {
    double alpha;
    double E;
    const system_data& sys;

    void operator()(const state& y, state& dy, double x) {
        dy[0] = y[1];

        if (x > global_eps) {
            dy[1] = ((pot(x, sys) - E) / alpha - 0.25 / (x * x)) * y[0];
        } else {
            dy[1] = 0;
        }
    }
};

struct wf_E_s {
    double lambda_s;
    const system_data& sys;
};

struct wf_gen_E_s {
    double rmin;
    double alpha;
    const system_data& sys;
};

