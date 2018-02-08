#pragma once
#include "common.h"
#include "templates.h"

/*
  Here are defined the series expansion approximation to the
  integrals, as opposed to computing the integrals numerically.
*/

/*** Analytic versions ***/

double analytic_prf(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);
double analytic_pr(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);

/*** Pole contribution: series expansion ***/

double fluct_e_tf(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n);

double fluct_es_f(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);

double fluct_e_tdf(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n);

double fluct_es_df(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);

double fluct_es_dfn(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);

/* Using Arb */

double fluct_e_tfr(double z, double E, double mr_ep, double mr_hp, double mu_e, double mu_h, uint32_t n);

/*** Pole contribution: pole position and residue ***/

double fluct_pf(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);

double fluct_pr(double a, double E, double mr_ep, double mr_hp, double mu_e, double mu_h);

/*** Pole contribution: momentum integral ***/

double fluct_pm(double a, double mr_ep, double mr_hp, double mu_e, double mu_h);

