#pragma once
#include "common.h"

struct plasmon_potcoef_s {
    double w;
    double k1;
    double k2;
    double mu_e;
    double mu_h;

    const system_data& sys;

    double delta{1e-12};
};

template <
    typename T,
    T (*potcoef_func)(
        const double wkk[3],
        double mu_e,
        double mu_h,
        const system_data& sys,
        double delta),
    bool is_green_complex = false>
struct plasmon_mat_s {
    using type = T;

    const uint32_t N_k;
    const uint32_t N_w;

    const system_data& sys;

    const double delta;

    arma::Row<T> row_G0;
    arma::Mat<T> mat_potcoef;
    arma::Mat<T> mat_elem;
    arma::Row<T> row_z_G0;

    arma::vec u0;
    arma::vec u1;
    double du0;
    double du1;

    void fill_row_G0() {
        arma::vec k_v{(1.0 - u0) / u0};

        arma::vec diag_vals{
            -0.5 * std::pow(sys.c_hbarc, 2) * arma::pow(k_v, 2) / sys.m_p,
        };

        arma::Row<T> diag_vals_t(N_k);

        if constexpr (std::is_same<T, std::complex<double>>::value) {
            diag_vals_t =
                arma::Row<T>(diag_vals.t().eval(), arma::zeros(N_k).t());
        } else {
            diag_vals_t = diag_vals.t().eval();
        }

        for (uint32_t i = 0; i < N_w; i++) {
            row_G0.cols(i * N_k, (i + 1) * N_k - 1) = diag_vals_t;
        }
    }

    void fill_mat_potcoef(double mu_e, double mu_h) {
        if constexpr (
            std::is_same<T, std::complex<double>>::value && is_green_complex) {
            /*
             * TODO: I think I can rewrite this part so that only
             * result is used, not potcoef.
             */
            arma::Cube<T> potcoef(N_k, N_k, N_w);

#pragma omp parallel for collapse(3)
            for (uint32_t i = 0; i < N_k; i++) {
                for (uint32_t j = 0; j < N_k; j++) {
                    for (uint32_t k = 0; k < N_w; k++) {
                        const double wkk[3] = {
                            u1(k) / (1.0 - std::pow(u1(k), 2)),
                            (1.0 - u0(i)) / u0(i),
                            (1.0 - u0(j)) / u0(j),
                        };

                        potcoef(i, j, k) =
                            potcoef_func(wkk, mu_e, mu_h, sys, delta);
                    }
                }
            }

            if (N_w > 1) {
                for (uint32_t i = 0; i < N_w; i++) {
                    for (uint32_t j = 0; j < i; j++) {
                        mat_potcoef.submat(
                            N_k * i, N_k * j, N_k * (i + 1) - 1,
                            N_k * (j + 1) - 1) = potcoef.slice(i - j);
                    }

                    for (uint32_t j = i; j < N_w; j++) {
                        mat_potcoef.submat(
                            N_k * i, N_k * j, N_k * (i + 1) - 1,
                            N_k * (j + 1) - 1) =
                            arma::conj(potcoef.slice(j - i));
                    }
                }
            } else {
                mat_potcoef = potcoef.slice(0);
            }

        } else {
#pragma omp parallel for
            for (uint32_t i = 0; i < N_k; i++) {
                for (uint32_t j = 0; j <= i; j++) {
                    const double wkk[3] = {
                        0.0,
                        (1.0 - u0(i)) / u0(i),
                        (1.0 - u0(j)) / u0(j),
                    };

                    T r = potcoef_func(wkk, mu_e, mu_h, sys, delta);

                    mat_potcoef(i, j) = r;
                    mat_potcoef(j, i) = r;
                }
            }
        }
    }

    void fill_mat_elem() {
        if (N_w > 1) {
#pragma omp parallel for collapse(4)
            for (uint32_t i = 0; i < N_w; i++) {
                for (uint32_t j = 0; j < N_w; j++) {
                    for (uint32_t k = 0; k < N_k; k++) {
                        for (uint32_t l = 0; l < N_k; l++) {
                            const double k_v[2]{
                                (1.0 - u0(k)) / u0(k),
                                (1.0 - u0(l)) / u0(l),
                            };

                            mat_elem(i + N_w * k, j + N_w * l) =
                                du0 * du1 * k_v[1] * (1 + std::pow(u1(j), 2)) /
                                std::pow(u0(l) * (1 - std::pow(u1(j), 2)), 2) *
                                mat_potcoef(i + N_w * k, j + N_w * l);
                        }
                    }
                }
            }
        } else {
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
    }

    template <typename T2>
    void update_mat_potcoef(T2 z) {
        row_z_G0.fill(-z);
        row_z_G0 += row_G0;
        row_z_G0 = 1.0 / row_z_G0;

        mat_potcoef.fill(arma::fill::eye);
        mat_potcoef += mat_elem.each_row() % row_z_G0;
    }

    plasmon_mat_s<T, potcoef_func>(
        uint32_t N_k,
        uint32_t N_w,
        const system_data& sys,
        double delta = 1e-12) :
        N_k(N_k),
        N_w(N_w),
        sys(sys),
        delta(delta),
        row_G0(arma::Row<T>(N_k * N_w)),
        mat_potcoef(arma::Mat<T>(N_k * N_w, N_k * N_w)),
        mat_elem(arma::Mat<T>(N_k * N_w, N_k * N_w)),
        row_z_G0(arma::Row<T>(N_k * N_w)),
        u0(arma::linspace(1.0 / N_k, 1 - 1.0 / N_k, N_k)),
        u1(arma::linspace(-1 + 1.0 / N_w, 1 - 1.0 / N_w, N_w)) {
        du0 = u0(1) - u0(0);
        if (N_w > 1) {
            du1 = u1(1) - u1(0);
        } else {
            du1 = std::numeric_limits<double>::quiet_NaN();
        }

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
struct plasmon_density_s {
    T& mat_s;

    double mu_e_lim{std::numeric_limits<double>::quiet_NaN()};
    double eb_lim{std::numeric_limits<double>::quiet_NaN()};

    double n_total{0};
};
