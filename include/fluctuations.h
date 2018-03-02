/*
 * This file contains definitions for the functions used
 * in computing the full solution to the semiconductor
 * problem: determining the chemical potential as a function
 * of the carrier density n.
 *
 * Since it is an extension of "analytic.h", some functions
 * used here will be defined in that file.
 */

#pragma once
#include "common.h"
#include "templates.h"
#include "analytic.h"

/*** Density ***/

/*
 * In this case it is not possible to compute each density
 * contribution using analytical expressions, thus forcing
 * us to use numerical methods to solve the integrals.
 *
 * In the following there are all the functions related
 * to computing the excitonic contribution.
 */

/*
 * Firstly, the many-body T matrix is used to compute the
 * spectral function. This integral is also called I_2(z).
 *
 * "fluct_T_z1" computes the value of T^{MB} at the point
 * z = z_1 = 1 / 4 * ( 1 - (m_p / m_m)^2 ) * E - mu_e - mu_h.
 * This point is special because for z > z1 it becomes
 * complex.
 *
 * "fluct_dT_dz" computes de derivative w. r. t. z.
 */

double fluct_T(double z, double E, double a, double mu_e, double mu_h, const system_data & sys);
double fluct_T_z1(double E, double a, double mu_e, double mu_h, const system_data & sys);
double fluct_dT_dz(double z, double E, double mu_e, double mu_h, const system_data & sys);

/*
 * As previously introduced, the many-body T matrix becomes
 * complex when z > z_1, and therefore a different set of
 * functions is required to find a solution.
 */

std::complex<double> fluct_T_c(double z, double E, double mu_e, double mu_h, const system_data & sys);

/*
 * These are the necessary ingredients to compute, first,
 * the Matsubara sum that results in the excitonic and
 * scattering contributions.
 *
 * For the excitonic contributions this results in a pole
 * for some frequency z = z_0, that is computed in the
 * following.
 */

/*
 * These two functions, "fluct_pp_*", compute the
 * position of the pole using either (s) Steffenson's
 * method or (b) Brent's method.
 *
 * "fluct_pp" is a helper function, to easily pick
 * between the two.
 */

double fluct_pp_s(double E, double a, double mu_e, double mu_h, const system_data & sys);
double fluct_pp_b(double E, double a, double mu_e, double mu_h, const system_data & sys);

template <bool brent = true, typename ... Args> double fluct_pp(Args ... args) {
  if constexpr(brent) {
    return fluct_pp_b(args...);
  } else {
    return fluct_pp_s(args...);
  }
}

/*
 * Because the behavior of the pole is not simple,
 * the following functions compute several different
 * important values related to the position of the
 * pole.
 *
 * "fluct_pp0_E" computes the value of E that satisfies
 * z_0(E, a) == 0. This is used later when computing
 * the final momentum integral over E, where one of
 * the bounds is given by this value.
 *
 * "fluct_pp0_a" computes the value of a that satisfies
 * z_0(E = 0, a) == 0. This value, also called ac_max,
 * is the maximum value of a that will not introduce
 * a singularity in the integrand of the momentum integral.
 *
 * "fluct_pp0_mu" is related to the previous. It gives
 * the value of mu_e (assuming equal ideal densities)
 * that satisfies fluct_pp0_a(mu_e) == a.
 */

double fluct_pp0_E(double a, double mu_e, double mu_h, const system_data & sys);
double fluct_pp0_a(double mu_e, double mu_h, const system_data & sys);
double fluct_pp0_mu(double a, double n, const system_data & sys);

/*
 * "fluct_pr" computes the residue of the pole. It takes
 * the value given by "fluct_pp" and multiplies it by the
 * appropiate factor.
 */

double fluct_pr(double E, double a, double mu_e, double mu_h, const system_data & sys);

/*
 * These are helping functions for finding special values
 * of the pole position.
 *
 * "fluct_ac" computes the critical scattering length for
 * a given energy E. For a > a_c(E) > a_c(0) the pole
 * disappears.
 *
 * "fluct_Ec" is the inverse of the previous function:
 * given a scattering length it computes the critical
 * energy at which the excitonic pole disappears.
 */

double fluct_ac(double E, double mu_e, double mu_h, const system_data & sys);
double fluct_Ec(double a, double mu_e, double mu_h, const system_data & sys);

/*
 * Now the momentum integral can be computed.
 *
 * "fluct_n_ex_c" checks that a > ac_max, in which case
 * there is a singularity in the integrand.
 */

double fluct_n_ex(double a, double mu_e, double mu_h, const system_data & sys);
double fluct_n_ex_c(double a, double ac_max, double mu_e, double mu_h, const system_data & sys);

/*
 * Branch contibution.
 *
 * In the following there are the functions required
 * for the computation of the scattering contribution
 * to the density.
 *
 * Firstly, solving the matsubara sum involves
 * solving an integral over the real axis.
 */

double fluct_bfi(double E, double a, double mu_e, double mu_h, const system_data & sys);

/*
 * Next, the momentum integral is computed from
 * "fluct_bfi".
 */

double fluct_n_sc(double a, double mu_e, double mu_h, const system_data & sys);

/*
 * Solving equation of state and self-consistency.
 */
std::vector<double> fluct_mu(double n, const system_data & sys);
std::vector<double> fluct_mu_steps(double n, std::vector<double> x_init, const system_data & sys);

