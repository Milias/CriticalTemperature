#pragma once
#include "common.h"

struct topo_potential_s {
    double k1;
    double k2;

    const system_data_v2& sys;
};

template <
    double (*potential_func)(const double[2], const system_data_v2&),
    arma::vec (*dispersion_func)(const arma::vec&, const system_data_v2&)>
struct topo_mat_s {
    using T = double;

    const uint32_t N_k;
    const system_data_v2& sys;

    arma::Row<T> row_G0;
    arma::Mat<T> mat_potcoef;
    arma::Mat<T> mat_elem;
    arma::Row<T> row_z_G0;

    arma::vec u0;
    double du0;
    double du1;

    void fill_row_G0() {
        /*
         * Computes the contribution from the free Hamiltonian,
         * with the given dispersion relation.
         */
        arma::vec k_v{(1.0 - u0) / u0};
        row_G0 = -dispersion_func(k_v, sys).t().eval();
    }

    void fill_mat_potcoef() {
        /*
         * Evaluates the matrix elements corresponding to the
         * interaction potential.
         */
#pragma omp parallel for
        for (uint32_t i = 0; i < N_k; i++) {
            for (uint32_t j = 0; j <= i; j++) {
                const double kk[2] = {
                    (1.0 - u0(i)) / u0(i),
                    (1.0 - u0(j)) / u0(j),
                };

                T r = potential_func(kk, sys);

                mat_potcoef(i, j) = r;
                mat_potcoef(j, i) = r;
            }
        }
    }

    void fill_mat_elem() {
        /*
         * Takes the interaction potential matrix elements and
         * adapts them to the discretization method.
         */
#pragma omp parallel for collapse(2)
        for (uint32_t i = 0; i < N_k; i++) {
            for (uint32_t k = 0; k < N_k; k++) {
                const double t_v[2] = {u0(i), u0(k)};

                const double k_v[2] = {
                    (1.0 - t_v[0]) / t_v[0],
                    (1.0 - t_v[1]) / t_v[1],
                };

                mat_elem(i, k) =
                    du0 * k_v[1] / std::pow(t_v[1], 2) * mat_potcoef(i, k);
            }
        }
    }

    void update_mat_potcoef(double z) {
        /*
         * The contribution G_0 = 1 / (z - H_0) is recomputed using
         * a different value for z.
         */
        row_z_G0.fill(-z);
        row_z_G0 += row_G0;
        row_z_G0 = 1.0 / row_z_G0;

        mat_potcoef.fill(arma::fill::eye);
        mat_potcoef += mat_elem.each_row() % row_z_G0;
    }

    topo_mat_s<potential_func, dispersion_func>(
        uint32_t N_k, const system_data_v2& sys) :
        N_k(N_k),
        sys(sys),
        row_G0(arma::Row<T>(N_k)),
        mat_potcoef(arma::Mat<T>(N_k, N_k)),
        mat_elem(arma::Mat<T>(N_k, N_k)),
        row_z_G0(arma::Row<T>(N_k)),
        u0(arma::linspace(1e-1 / N_k, 1 - 1e-1 / N_k, N_k)) {
        du0 = u0(1) - u0(0);
        fill_row_G0();
    }
};

struct plasmon_rpot_s {
    double x;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

struct plasmon_exc_mu_zero_s {
    const system_data& sys;

    double val{0.0};
};

template <typename T>
struct plasmon_exc_mu_lim_s {
    T& mat_s;

    double val{0};
    double eb_lim{std::numeric_limits<double>::quiet_NaN()};
};

template <typename T>
struct plasmon_exc_mu_lim_int_s {
    T& mat_s;
    double u;
    double val;

    double eb_lim{std::numeric_limits<double>::quiet_NaN()};
};

template <typename T>
struct plasmon_density_s {
    T& mat_s;

    double mu_e_lim{std::numeric_limits<double>::quiet_NaN()};
    double eb_lim{std::numeric_limits<double>::quiet_NaN()};

    double n_total{0};
};
