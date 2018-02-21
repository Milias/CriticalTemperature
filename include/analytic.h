/*
 * In this file we define the functions related to the
 * analytic solution of the semiconductor, neglecting
 * fluctuations.
 *
 * Here we also include ideal gas results, which will
 * be used in other parts of the program.
 *
 * Since in this case we have analytic expressions for
 * the density, we start from here and then move on to
 * solve the equations of state using the previous results.
 */

#pragma once

#include "common.h"
#include "templates.h"

/*** Density ***/

/*
 * mu_i is the chemical potential of one species.
 *
 * m_pi is the ratio between the reduced mass and
 * the mass of the species.
 *
 * m_sigma is 1/m_pe + 1/m_ph.
 *
 * a is one over the scattering length.
 *
 * ideal_n is the density of an ideal gas with
 * chemical potential mu_i and mass m_pi.
 *
 * analytic_n_ex is the excitonic contribution to
 * the total density, which only depends on the total
 * chemical potential, scattering length and m_sigma
 * as defined above.
 *
 * analytic_n_sc is the scattering contribution
 * to the total density, depends on the same
 * parameters as the previous one.
 */

double ideal_n(double mu_i, double m_pi);
double analytic_n_ex(double mu_t, double a, const system_data & sys);
double analytic_n_sc(double mu_t, double a, const system_data & sys);

/*** Chemical potential ***/

/*
 * First we need to define several functions
 * to compute the chemical potential for the
 * ideal gas in terms of different parameters.
 *
 * ideal_mu computes the chemical potential
 * of an ideal gas of density n and mass ratio
 * m_pa.
 *
 * ideal_mu_dn is the derivative of the
 * chemical potential w. r. t. the density n.
 *
 * ideal_mu_h computes the chemical potential
 * of the holes from the electrons assuming that their
 * densities as ideal gases are the same, that is:
 *
 * n_id,e == n_id,h.
 *
 * The arguments are: mu_a the chemical potential
 * of the first species, m_pb the mass ratio of
 * the other species and m_pa the mass ratio of the
 * first.
 *
 * ideal_mu_v computes the value of the
 * chemical potential for one species assuming
 * same ideal densities that satisfies:
 *
 * mu_e + mu_h == v,
 *
 * with mu_0 the initial guess for the solution,
 * and m_pe, m_ph the reduced mass over each
 * species' masses.
 *
 * ideal_lambda_s computes Yukawa's length scale
 * for a given ideal electron density n_id, assuming
 * that the main contributors are free electrons
 * and holes.
 *
 * NOTE: notice that n_id is not the full carrier
 * density n, only electrons and holes contribute.
 */

double ideal_mu(double n, double m_pi);
double ideal_mu_dn(double n, double m_pi);
double ideal_mu_h(double mu_e, const system_data & sys);
double ideal_mu_v(double v, double mu_0, const system_data & sys);
double ideal_ls(double n_id, const system_data & sys);

/*
 * As a part of the self-consistent procedure we follow
 * to compute the chemical potential in terms of the
 * carrier density, we need to compute the scattering
 * length a in terms of, ultimately, the chemical potential.
 *
 * analytic_a_ls takes a value ls for Yukawa's potential
 * length scale and solves the ODE that returns the scattering
 * length a that we need.
 *
 * For more information check arXiv:1505.01732, section II.B.
 */
double analytic_a_ls(double ls, const system_data & sys);

/*
 * Here we compute the chemical potential for electrons
 * assuming same ideal densities for a given carrier
 * density n, and with a self-consistently computed
 * scattering length.
 *
 * The argument m_pT is the ratio of m_p and energy_th:
 *
 * energy_th = 1 / (4 * M_PI * beta)
 */
std::vector<double> analytic_mu(double n, const system_data & sys);

