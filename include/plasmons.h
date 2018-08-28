#pragma once
#include "common.h"
#include "templates.h"

std::vector<double> plasmon_green(double w, double k, double mu_e, double mu_h, const system_data & sys, double delta = 1e-6);

std::vector<double> plasmon_potcoef(double w, double k1, double k2, double mu_e, double mu_h, const system_data & sys);

std::vector<double> plasmon_sysmatelem(uint32_t i, uint32_t j, uint32_t k, uint32_t l, std::complex<double> z, double dk, std::complex<double> elem);

