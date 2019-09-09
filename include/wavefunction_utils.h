#pragma once

#include "common.h"

struct wf_E_s {
    double lambda_s;
    const system_data& sys;
};

struct wf_bexc_E_s {
    const system_data& sys;
    double lj_param_a{1.0};
    double lj_param_b{1.0};
};

