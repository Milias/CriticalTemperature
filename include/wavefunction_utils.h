#pragma once

#include "common.h"

template <typename state, class pot_s>
struct wf_dy_s {
    double alpha;
    double E;
    pot_s& pot;

    void operator()(const state& y, state& dy, double x) {
        dy[0] = y[1];

        if (x > global_eps) {
            dy[1] = ((pot(x) - E) / alpha - 0.25 / (x * x)) * y[0];
        } else {
            dy[1] = 0;
        }
    }
};

struct wf_E_s {
    double lambda_s;
    const system_data& sys;
};

template <class pot_s>
struct wf_gen_E_s {
    double rmin;
    double alpha;
    pot_s& pot;
    double rmax{0.0};
};

