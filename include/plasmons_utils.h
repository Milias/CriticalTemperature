#pragma once
#include "common.h"

struct plasmon_potcoef_s {
  double w;
  double k1;
  double k2;
  double mu_e;
  double mu_h;

  const system_data & sys;

  double delta{1e-3};
};

