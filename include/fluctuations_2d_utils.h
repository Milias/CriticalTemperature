/*
 * Here we define several helper structures
 * used in fluctuations_2d.cpp.
 */

#pragma once
#include "common.h"

struct fluct_2d_I2_s {
  double z;
  double E;
  double mu_e;
  double mu_h;

  const system_data & sys;
};
