#pragma once
#include "common.h"

struct biexciton_eqst_s {
    double n_gamma;
    double be_exc;
    double be_biexc;
    const system_data& sys;

    double mu_e_lim;
    double u{0.0};
};

struct biexciton_pot_s {
    double param_alpha;
    double param_th1{0.0};
    double param_x1{0.0};
    double param_x2{0.0};
};

template <class pot_s>
struct biexciton_rmin_s {
    double E_min;
    double be_biexc;

    pot_s& pot;
    const system_data& sys;
};

template <class pot_s>
struct biexciton_c12_s {
    double be_biexc;

    pot_s& pot;
    const system_data& sys;
};
