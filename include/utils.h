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

    constexpr static double c_kB{8.6173303e-5};             // eV K^-1
    constexpr static double c_m_e{0.5109989461e6};          // eV
    constexpr static double c_hbar{6.582119514e-16};        // eV s
    constexpr static double c_light{299792458e9};           // nm s^-1
    constexpr static double c_hbarc{1.9732697879518254e2};  // eV nm
    constexpr static double c_e_charge{1.602176620898e-19}; // C
    /*
     * Electromagnetic fine-structure constant,
     * approx ~ 1 / 137.
     */
    constexpr static double c_aEM{7.2973525664e-3}; // dimensionless

    /*
     * Basic system_data constants
     */

    const double m_e;   // eV
    const double m_h;   // eV
    const double eps_r; // dimensionless

    /*
     * Original constructor arguments
     */

    const double dl_m_e;
    const double dl_m_h;

    /*
     * Derived and dimensionless constants
     */

    const double m_p;          // eV
    const double m_2p;         // eV
    const double m_pe;         // dimensionless
    const double m_ph;         // dimensionless
    const double m_sigma;      // dimensionless
    const double m_eh;         // dimensionless
    const double c_alpha;      // eV nm^2
    const double c_alpha_bexc; // eV nm^2 but for biexctions
    const double a0;           // nm

    /*
     * Temperature-scaled quantities
     *
     * Changes to these should be done
     * through the method system_data::set_temperature(double T).
     *
     * E_1 is the groundstate energy given a
     * Coulomb interaction.
     */

    double T;               // K
    double beta;            // eV^-1
    double lambda_th;       // nm
    double lambda_th_biexc; // nm
    double energy_th;       // eV
    double m_pT;            // dimensionless
    double E_1;             // dimensionless
    double delta_E;         // dimensionless
    double sys_ls;          // nm^-1

    /*
     * Temperature-less units
     *
     */

    const double zt_len;
    double e_F{0}; // eV
    double k_F{0}; // nm^-1

    double E0{0}; // eV
    double l0{0}; // nm

    double eps_mat{1.0};    // dimensionless
    double size_d{0.0};     // nm
    double size_Lx{0.0};    // nm
    double size_Ly{0.0};    // nm
    double hwhm_x{0.0};     // nm
    double hwhm_y{0.0};     // nm
    double sigma_x{0.0};    // nm
    double sigma_y{0.0};    // nm
    double ext_dist_l{0.0}; // nm

    /*
     * Constructor methods
     *
     * m_e and m_h here are dimensionless!
     * They will be multiplied by c_m_e.
     */
    system_data(
        double m_e,
        double m_h,
        double eps_r,
        double T,
        double size_d     = 1.0,
        double size_Lx    = 0.0,
        double size_Ly    = 0.0,
        double hwhm_x     = 0.0,
        double hwhm_y     = 0.0,
        double eps_mat    = 1.0,
        double ext_dist_l = 0.0);
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
    constexpr double f_beta(double T) { return 1.0 / (c_kB * T); }

    /*
     * Computes de Broglie's thermal wavelength
     * from beta and reduced mass:
     *
     * lambda_th = c * hbar * sqrt(2 * pi * beta / m_p)
     */
    double f_lambda_th(double beta, double m_p) {
        return c_hbarc * std::sqrt(2 * M_PI * beta / m_p);
    }

    /*
     * Computes the thermal energy from beta:
     *
     * energy_th = 1 / ( 4 * pi * beta )
     */
    constexpr double f_energy_th(double beta) { return 0.25 * M_1_PI / beta; }

    /*
     * Use this method when changing the temperature,
     * it will recompute the related quantities.
     */
    void set_temperature(double T);

    /*
     * Compute the value of z_1 given the energy
     * and chemical potential given.
     */
    double get_z1(double E, double mu_t) const;

    /*
     * Compute the energy level with a given
     * quantum number n assuming Coulomb
     * potential.
     */
    template <uint32_t n = 1>
    double get_E_n() const {
        return E_1 / (n * n);
    }

    double get_E_n(double n) const;

    double get_mu_h_t0(double mu_e) const;
    double get_mu_h_ht(double mu_e) const;
    double get_mu_h(double mu_e) const;

    double distr_fd(double energy, double mu) const;
    double distr_be(double energy, double mu) const;
    double distr_mb(double energy, double mu) const;

    double density_ideal_t0(double mu_e) const;
    double density_ideal_ht(double mu_e) const;
    double density_ideal(double mu_e) const;

    double density_exc_ht(double mu_ex, double eb) const;
    double density_exc_exp(double u) const;
    double density_exc_exp_ht(double u) const;
    double density_exc(double mu_ex, double eb) const;

    double density_exc2(double mu_ex, double eb_ex, double eb_ex2) const;
    double density_exc2_u(double u) const;

    double mu_ideal(double n) const;
    double mu_h_ideal(double n) const;
    double mu_exc_u(double n) const;

    double ls_ideal(double n) const;

    double exc_mu_zero() const;
    double exc_mu_val(double val) const;

    double exc_bohr_radius() const;
    double exc_bohr_radius_mat() const;

    double eta_func() const;

    void set_hwhm(double hwhm_x, double hwhm_y);
};

struct sys_params {
    /*
     * Mass.
     * NOTE: Proportional to c_m_e!!
     */
    double m_e;
    double m_hh;
    double m_lh;

    /*
     * Environment.
     */
    double eps_sol; // dimensionless
    double T;       // K
    double size_d;  // nm

    /*
     * Topological hamiltonian parameters.
     */
    double M, A1, A2, B1, B2, C, D1, D2;

    sys_params()                         = default;
    sys_params(const sys_params& params) = default;
};

struct der_params {
    /*
     * By default, m_p = m_p_hh.
     */
    double m_p; // eV
    double m_pe;
    double m_phh;
    double beta; // eV^-1

    der_params(const sys_params& params) :
        m_p(system_data::c_m_e * params.m_e * params.m_hh /
            (params.m_e + params.m_hh)),
        m_pe(m_p / params.m_e / system_data::c_m_e),
        m_phh(m_p / params.m_hh / system_data::c_m_e),
        beta(1 / (system_data::c_kB * params.T)) {}
};

struct system_data_v2 {
    /*
     * Fundamental constants
     */

    constexpr static double c_kB{8.6173303e-5};             // eV K^-1
    constexpr static double c_m_e{0.5109989461e6};          // eV
    constexpr static double c_hbar{6.582119514e-16};        // eV s
    constexpr static double c_light{299792458e9};           // nm s^-1
    constexpr static double c_hbarc{1.9732697879518254e2};  // eV nm
    constexpr static double c_e_charge{1.602176620898e-19}; // C
    /*
     * Electromagnetic fine-structure constant,
     * approx ~ 1 / 137.
     */
    constexpr static double c_aEM{7.2973525664e-3}; // dimensionless

    /*
     * Other parameters.
     */

    sys_params params;
    der_params d_params;

    /*
     * Original constructor arguments
     */
    double m_e;
    double m_hh;
    double m_lh;

    /*
     * Constructor methods
     */
    system_data_v2(const sys_params& params);
    system_data_v2(const system_data_v2& sys) = default;

    ~system_data_v2() {}

    /*
     * Chemical potentials.
     */

    double mu_hh_t0(double mu_e) const;
    double mu_hh_ht(double mu_e) const;
    double mu_hh(double mu_e) const;

    // Computes mu_e such that mu_e + mu_h == val.
    double exc_mu_val(double val) const;

    // Computes chemical potential from densities.
    double mu_ideal(double n) const;
    double mu_h_ideal(double n) const;
    double mu_exc(double n, double E_X) const;

    /*
     * Densities
     */
    double density_ideal(double mu_e) const;
    double density_exc(double mu_ex, double eb) const;
};
