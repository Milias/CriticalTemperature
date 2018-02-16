#pragma once
#include "common.h"
#include "templates.h"

/*** Analytic versions ***/

double analytic_prf(double a, double E, double m_pe, double m_ph, double mu_e, double mu_h);
double analytic_pr(double a, double E, double m_pe, double m_ph, double mu_e, double mu_h);

/*** T^{MB} ***/

double fluct_t(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h, double a);
double fluct_t_z1(double E, double m_pe, double m_ph, double mu_e, double mu_h, double a);
double fluct_t_dtdz(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Pole contribution: fluctuations integral ***/

double fluct_i(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_i_z1(double E, double m_pe, double m_ph, double mu_e, double mu_h);

double fluct_i_dfdz_n(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Pole contribution: pole position and pole residue ***/

double fluct_pp(double a, double E, double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_pp_b(double a, double E, double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_pp0(double a, double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_pp0c(double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_pp0c_mu(double a, double n, double m_pe, double m_ph);

double fluct_pr(double a, double E, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Pole contribution: momentum integral ***/

double fluct_pmi_nc(double a, double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_pmi(double a, double ac_max, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Critical parameters: a_c(E) and E_c(a) ***/

double fluct_ac(double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_ac_E(double E, double m_pe, double m_ph, double mu_e, double mu_h);
double fluct_Ec_a(double a, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Branch contribution: I_2(z) ***/

std::vector<double> fluct_i_c_fbv(double z, double E, double mr_ip, double mu_t);
std::complex<double> fluct_i_c_fc(double x, double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);
std::complex<double> fluct_i_c(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);
std::complex<double> fluct_i_ci(double z, double E, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Branch contribution: matsubara sum ***/

double fluct_bfi_spi(double y, double E, double m_pe, double m_ph, double mu_e, double mu_h, double a);

double fluct_bfi(double E, double m_pe, double m_ph, double mu_e, double mu_h, double a);

/*** Branch contribution: momentum integral ***/

double fluct_bmi(double a, double m_pe, double m_ph, double mu_e, double mu_h);

/*** Chemical potential: fixed scattering length ***/

double fluct_mu_a_fp(double mu_e, double n, double m_pe, double m_ph, double a);
std::vector<double> fluct_mu_a(double n, double a, double m_pe, double m_ph);
std::vector<double> fluct_mu_a_s(double n, double a, double m_pe, double m_ph);

