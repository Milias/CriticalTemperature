#pragma once
#include "common.h"

template <class pot_s>
struct exciton_root_s {
    double be_exc;

    pot_s& pot;
    const system_data& sys;
};

struct exciton_PL_s {
    double energy;

    uint32_t nx;
    uint32_t ny;

    const system_data& sys;

    double gamma = 0.0;
    double t = 0.0;
};
