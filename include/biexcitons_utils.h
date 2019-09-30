#pragma once
#include "common.h"

struct biexciton_pot_s {
    double param_alpha;
    double param_th1{0.0};
    double param_x1{0.0};
    double param_x2{0.0};
};

template <class pot_s>
struct biexciton_rmin_s {
    double E_min;
    double eb_biexc;

    pot_s& pot;
    const system_data& sys;
};
