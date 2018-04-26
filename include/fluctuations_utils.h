/*
 * Here we define several helper structures
 * used in fluctuations.cpp.
 */

#pragma once
#include "common.h"

struct fluct_T_i_s {
  constexpr static double c_f{- M_SQRT1_2 / 4.0};

  double z;
  double E;
  double mu_e;
  double mu_h;
  const system_data & sys;
};

struct fluct_T_z1_i_s {
  constexpr static double c_f{- M_SQRT1_2 / 4.0};

  double E;
  double mu_e;
  double mu_h;
  const system_data & sys;
};

struct fluct_T_i_c_s {
  constexpr static double c_f{- M_SQRT1_2 / 4.0};

  double z;
  double E;
  double mu_e;
  double mu_h;
  const system_data & sys;
};

struct fluct_pp_s {
  double E;
  double a;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_pp0_E_s {
  double a;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_pp0_a_s {
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_pp0_mu_s {
  double a;

  const system_data & sys;
};

struct fluct_Ec_s {
  double a;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_n_ex_s {
  constexpr static double c_f{1 / (16 * M_PI * M_PI)};

  double a;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_bfi_s {
  double E;
  double a;
  double mu_e;
  double mu_h;
  double z1;

  const system_data & sys;
};

struct fluct_n_sc_s {
  constexpr static double c_f{-1 / ( 32 * M_PI * M_PI * M_PI)};

  double a;
  double mu_e;
  double mu_h;

  const system_data & sys;
};

struct fluct_mu_s {
  double n;

  const system_data & sys;
};

