#pragma once

#include "common.h"

/*
 * In this file there are several structures and other necessary
 * components that will be used throughout the rest of the program.
 */

struct system_data {
  /*
   * This structure contains all the possible variables needed
   * for every function in the program.
   *
   * Since the mass of the electron and holes, temperature and
   * relative permitivity are the only parameters set by external
   * factors, those will set some of the others.
   *
   * Here there are also defined some of the fundamental constants
   * needed for this purpose.
   */

  /*
   * Fundamental constants
   */

  constexpr static double c_kB{8.6173303e-5}; // eV K^-1
  constexpr static double c_m_e{0.5109989461e6}; // eV
  constexpr static double c_hbar{6.582119514e-16}; // eV s
  constexpr static double c_light{299792458}; // m s^-1
  /*
   * Electromagnetic fine-structure constant,
   * approx ~ 1 / 137.
   */
  constexpr static double c_aEM{7.2973525664e-3}; // dimensionless

  /*
   * Basic system_data constants
   */

  const double m_e; // eV
  const double m_h; // eV
  const double eps_r; // dimensionless

  /*
   * Original constructor arguments
   */

  const double dl_m_e;
  const double dl_m_h;

  /*
   * Derived and dimensionless constants
   */

  const double m_p; // eV
  const double m_pe; // dimensionless
  const double m_ph; // dimensionless
  const double m_sigma; // dimensionless

  /*
   * Temperature-scaled quantities
   *
   * Changes to these should be done
   * through the method system_data::set_temperature(double T).
   */

  double T; // K
  double beta; // eV^-1
  double lambda_th; // m
  double energy_th; // eV
  double m_pT; // dimensionless

  /*
   * Constructor methods
   *
   * m_e and m_h here are dimensionless!
   * They will be multiplied by c_m_e.
   */
  system_data(double m_e, double m_h, double eps_r, double T);
  system_data(const system_data& sys) = default;

  ~system_data() {}

  /*
   * Member functions
   */

  /*
   * Computes beta from the temperature:
   *
   * beta = 1 / ( c_kB * T)
   */
  constexpr double f_beta(double T) {
    return 1.0 / (c_kB * T);
  }

  /*
   * Computes de Broglie's thermal wavelength
   * from beta and reduced mass:
   *
   * lambda_th = c * hbar * sqrt(2 * pi * beta / m_p)
   */
  double f_lambda_th(double beta, double m_p) {
    return c_light * c_hbar * std::sqrt(2 * M_PI * beta / m_p);
  }

  /*
   * Computes the thermal energy from beta:
   *
   * energy_th = 1 / ( 4 * pi * beta )
   */
  constexpr double f_energy_th(double beta) {
    return 0.25 * M_1_PI / beta;
  }

  /*
   * Use this method when changing the temperature,
   * it will recompute the related quantities.
  */
  void set_temperature(double T);
};

