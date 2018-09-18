#pragma once
#include "common.h"
#include "templates.h"

std::vector<double> plasmon_green(double w, double k, double mu_e, double mu_h, double v_1, const system_data & sys, double delta = 1e-6);

double plasmon_kmax(double mu_e, double mu_h, double v_1, const system_data & sys);
double plasmon_wmax(double mu_e, double mu_h, double v_1, const system_data & sys);

double plasmon_disp(double k, double mu_e, double mu_h, double v_1, const system_data & sys);
double plasmon_disp_ncb(double k, double mu_e, double mu_h, double v_1, const system_data & sys, double kmax);

double plasmon_disp_inv(double w, double mu_e, double mu_h, double v_1, const system_data & sys);
double plasmon_disp_inv_ncb(double w, double mu_e, double mu_h, double v_1, const system_data & sys);

std::vector<double> plasmon_potcoef(const std::vector<double> & wkk, double mu_e, double mu_h, double v_1, const system_data & sys, double delta = 1e-2);

std::vector<double> plasmon_sysmatelem(const std::vector<uint32_t> & ids, const std::vector<double> & wkwk, double dw, double dk, const std::complex<double> & z, const std::complex<double> & elem, const system_data & sys);

