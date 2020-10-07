#pragma once
#include "common.h"

template <
    typename topo_functor,
    arma::vec (*dispersion_func)(const arma::vec&, const system_data_v2&)>
struct topo_mat_s {
    using T = double;

    const uint32_t N_k;
    topo_functor pot_s;

    arma::Row<T> row_G0;
    arma::Mat<T> mat_potcoef;
    arma::Mat<T> mat_elem;
    arma::Row<T> row_z_G0;

    arma::vec u0;
    double du0;
    double du1;

    double pot_integral(const double kk[2]) {
        constexpr uint32_t n_int{1};
        double result[n_int] = {0}, error[n_int] = {0};

        topo_functor n_pot_s{pot_s.set_momentum(kk)};

        gsl_function integrands[n_int];

        integrands[0].function = &functor_call<topo_functor>;
        integrands[0].params   = &n_pot_s;

        constexpr uint32_t local_ws_size{(1 << 7)};
        constexpr double local_eps{1e-8};

        gsl_integration_workspace* ws =
            gsl_integration_workspace_alloc(local_ws_size);

        gsl_integration_qag(
            integrands, 2 * M_PI, 0, local_eps, 0, local_ws_size,
            GSL_INTEG_GAUSS31, ws, result, error);

        gsl_integration_workspace_free(ws);

        return result[0] * M_1_PI * 0.5;
    }

    void fill_row_G0() {
        /*
         * Computes the contribution from the free Hamiltonian,
         * with the given dispersion relation.
         */
        arma::vec k_v{(1.0 - u0) / u0};
        row_G0 = -dispersion_func(k_v, pot_s.sys).t().eval();
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

                T r = pot_integral(kk);

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
                    //du0 * k_v[1] / std::pow(t_v[1], 2) * mat_potcoef(i, k);
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

    topo_mat_s<topo_functor, dispersion_func>(
        uint32_t N_k, topo_functor pot_s) :
        N_k(N_k),
        pot_s(pot_s),
        row_G0(arma::Row<T>(N_k)),
        mat_potcoef(arma::Mat<T>(N_k, N_k)),
        mat_elem(arma::Mat<T>(N_k, N_k)),
        row_z_G0(arma::Row<T>(N_k)),
        u0(arma::linspace(1e-1 / N_k, 1 - 1e-1 / N_k, N_k)) {
        du0 = u0(1) - u0(0);
        fill_row_G0();
    }
};

