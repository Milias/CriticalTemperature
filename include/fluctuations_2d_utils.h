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

struct fluct_2d_I2_p_s {
  double x_0e;
  double dx_0e;
  double x_me;
  double x_pe;
  double ye_max;

  double x_0h;
  double dx_0h;
  double x_mh;
  double x_ph;
  double yh_max;

  double sign_e;
  double sign_h;

  double E;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_2d_pp_s {
  double z1;

  double E;
  double chi_ex;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_2d_n_ex_s {
  double chi_ex;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_2d_n_sc_s {
  double t;
  double z;

  double chi_ex;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

