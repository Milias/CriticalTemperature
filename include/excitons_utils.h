#pragma once
#include "common.h"

template <class pot_s>
struct exciton_root_s {
    double be_exc;

    pot_s& pot;
    const system_data& sys;
};

struct exciton_lorentz_s {
    double energy;

    uint32_t nx;
    uint32_t ny;

    const system_data& sys;

    double gamma = 0.0;
    double t     = 0.0;
    double th    = 0.0;
};

struct exciton_cont_s {
    double energy;
    double gamma_c;

    const system_data& sys;

    double y = 0.0;
};
