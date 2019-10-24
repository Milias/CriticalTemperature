#pragma once
#include "common.h"

template <class pot_s>
struct exciton_root_s {
    double be_exc;

    pot_s& pot;
    const system_data& sys;
};
