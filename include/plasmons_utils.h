#pragma once
#include "common.h"

struct plasmon_kmax_s {
  double mu_e;
  double mu_h;
  double v_1;

  const system_data & sys;
};

struct plasmon_disp_s {
  double k;
  double mu_e;
  double mu_h;
  double v_1;

  const system_data & sys;
};

struct plasmon_disp_inv_s {
  double w;
  double mu_e;
  double mu_h;
  double v_1;

  const system_data & sys;
};

struct plasmon_potcoef_s {
  double w;
  double k1;
  double k2;
  double mu_e;
  double mu_h;
  double v_1;

  const system_data & sys;

  double delta{1e-2};
};

