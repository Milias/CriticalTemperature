#include "topo.h"
#include "topo_utils.h"

double topo_green_cou(double k, const system_data_v2& sys) {
    return -sys.c_hbarc * sys.c_aEM / (sys.params.eps_sol * k);
}

arma::cx_mat44 topo_ham_3d(
    double kx, double ky, double kz, const system_data_v2& sys) {
    arma::cx_mat44 hamilt0(arma::fill::none);

    const double k2{kx * kx + ky * ky};
    const double kz2{kz * kz};
    const std::complex<double> kp(kx, ky);
    const std::complex<double> km(kx, -ky);

    hamilt0(0, 0) = sys.params.C - (sys.params.B2 - sys.params.D2) * k2 +
                    (-sys.params.B1 + sys.params.D1) * kz2 + sys.params.M;
    hamilt0(1, 0) = sys.params.A1 * kz;
    hamilt0(2, 0) = 0;
    hamilt0(3, 0) = sys.params.A2 * km;

    hamilt0(0, 1) = sys.params.A1 * kz;
    hamilt0(1, 1) = sys.params.C + (sys.params.B2 + sys.params.D2) * k2 +
                    (sys.params.B1 + sys.params.D1) * kz2 - sys.params.M;
    hamilt0(2, 1) = sys.params.A2 * km;
    hamilt0(3, 1) = 0;

    hamilt0(0, 2) = 0;
    hamilt0(1, 2) = sys.params.A2 * kp;
    hamilt0(2, 2) = sys.params.C - (sys.params.B2 - sys.params.D2) * k2 +
                    (-sys.params.B1 + sys.params.D1) * kz2 + sys.params.M;
    hamilt0(3, 2) = -sys.params.A1 * kz;

    hamilt0(0, 3) = sys.params.A2 * kp;
    hamilt0(1, 3) = 0;
    hamilt0(2, 3) = -sys.params.A1 * kz;
    hamilt0(3, 3) = sys.params.C + (sys.params.B2 + sys.params.D2) * k2 +
                    (sys.params.B1 + sys.params.D1) * kz2 - sys.params.M;

    return hamilt0;
}

template <uint32_t n = 4>
arma::vec::fixed<n> topo_eigenval_3d(
    double k, double kz, const system_data_v2& sys) {
    arma::vec::fixed<n> vals(arma::fill::none);

    const double k2{k * k}, kz2{kz * kz};

    vals(0) =
        sys.params.C + sys.params.D2 * k2 + sys.params.D1 * kz2 -
        std::sqrt(
            std::pow(sys.params.A2, 2) * k2 +
            std::pow(sys.params.A1, 2) * kz2 +
            std::pow(
                sys.params.B2 * k2 + sys.params.B1 * kz2 - sys.params.M, 2));
    vals(1) =
        sys.params.C + sys.params.D2 * k2 + sys.params.D1 * kz2 +
        std::sqrt(
            std::pow(sys.params.A2, 2) * k2 +
            std::pow(sys.params.A1, 2) * kz2 +
            std::pow(
                sys.params.B2 * k2 + sys.params.B1 * kz2 - sys.params.M, 2));

    if constexpr (n == 4) {
        vals(2) = vals(1);
        vals(3) = vals(1);
        vals(1) = vals(0);
    }

    return vals;
}

template <uint32_t n = 4>
arma::vec::fixed<n> topo_eigenval_2d(double k, const system_data_v2& sys) {
    arma::vec::fixed<n> vals(arma::fill::none);

    const double k2{k * k};

    vals(0) = sys.params.C + sys.params.D2 * k2 -
              std::sqrt(
                  std::pow(sys.params.A2, 2) * k2 +
                  std::pow(sys.params.B2 * k2 - sys.params.M, 2));
    vals(1) = sys.params.C + sys.params.D2 * k2 +
              std::sqrt(
                  std::pow(sys.params.A2, 2) * k2 +
                  std::pow(sys.params.B2 * k2 - sys.params.M, 2));

    if constexpr (n == 4) {
        vals(2) = vals(1);
        vals(3) = vals(1);
        vals(1) = vals(0);
    }

    return vals;
}

arma::cx_mat44 topo_eigenvec_3d(
    double kx, double ky, double kz, const system_data_v2& sys) {
    const double k2{kx * kx + ky * ky};
    if (k2 < 1e-14) {
        return arma::cx_mat44(arma::fill::eye);
    }

    arma::cx_mat44 vecs(arma::fill::none);

    const double kz2{kz * kz};
    const std::complex<double> kp(kx, ky);
    const std::complex<double> km(kx, -ky);

    vecs(0, 0) =
        -(sys.params.B2 * k2 + sys.params.B1 * kz2 +
          std::sqrt(
              std::pow(sys.params.A2, 2) * k2 +
              std::pow(sys.params.A1, 2) * kz2 +
              std::pow(
                  sys.params.B2 * k2 + sys.params.B1 * kz2 - sys.params.M,
                  2)) -
          sys.params.M) /
        (sys.params.A2 * kp);
    vecs(1, 0) = sys.params.A1 * kz / (sys.params.A2 * kp);
    vecs(2, 0) = 0;
    vecs(3, 0) = 1;

    vecs(0, 2) = sys.params.A1 * kz / (sys.params.A2 * kp);
    vecs(1, 2) =
        (sys.params.B2 * k2 + sys.params.B1 * kz2 -
         std::sqrt(
             std::pow(sys.params.A2, 2) * k2 +
             std::pow(sys.params.A1, 2) * kz2 +
             std::pow(
                 sys.params.B2 * k2 + sys.params.B1 * kz2 - sys.params.M, 2)) -
         sys.params.M) /
        (sys.params.A2 * kp);
    vecs(2, 2) = 1;
    vecs(3, 2) = 0;

    vecs(0, 3) =
        (-sys.params.B2 * k2 - sys.params.B1 * kz2 +
         std::sqrt(
             std::pow(sys.params.A2, 2) * k2 +
             std::pow(sys.params.A1, 2) * kz2 +
             std::pow(
                 sys.params.B2 * k2 + sys.params.B1 * kz2 - sys.params.M, 2)) +
         sys.params.M) /
        (sys.params.A2 * kp);
    vecs(1, 3) = sys.params.A1 * kz / (sys.params.A2 * kp);
    vecs(2, 3) = 0;
    vecs(3, 3) = 1;

    vecs(0, 1) = sys.params.A1 * kz / (sys.params.A2 * kp);
    vecs(1, 1) =
        (sys.params.B2 * k2 + sys.params.B1 * kz2 +
         std::sqrt(
             std::pow(sys.params.A2, 2) * k2 +
             std::pow(sys.params.A1, 2) * kz2 +
             std::pow(
                 sys.params.B2 * k2 + sys.params.B1 * kz2 - sys.params.M, 2)) -
         sys.params.M) /
        (sys.params.A2 * kp);
    vecs(2, 1) = 1;
    vecs(3, 1) = 0;

    return vecs;
}

arma::cx_mat44 topo_eigenvec_2d(
    double kx, double ky, const system_data_v2& sys) {
    const double k2{kx * kx + ky * ky};
    if (k2 < 1e-14) {
        return arma::cx_mat44(arma::fill::eye);
    }

    arma::cx_mat44 vecs(arma::fill::none);

    const std::complex<double> kp(kx, ky);
    const std::complex<double> km(kx, -ky);

    vecs(0, 0) = -(sys.params.B2 * k2 +
                   std::sqrt(
                       std::pow(sys.params.A2, 2) * k2 +
                       std::pow(sys.params.B2 * k2 - sys.params.M, 2)) -
                   sys.params.M) /
                 (sys.params.A2 * kp);
    vecs(1, 0) = 0;
    vecs(2, 0) = 0;
    vecs(3, 0) = 1;

    vecs(0, 2) = 0;
    vecs(1, 2) = (sys.params.B2 * k2 -
                  std::sqrt(
                      std::pow(sys.params.A2, 2) * k2 +
                      std::pow(sys.params.B2 * k2 - sys.params.M, 2)) -
                  sys.params.M) /
                 (sys.params.A2 * kp);
    vecs(2, 2) = 1;
    vecs(3, 2) = 0;

    vecs(0, 3) = (-sys.params.B2 * k2 +
                  std::sqrt(
                      std::pow(sys.params.A2, 2) * k2 +
                      std::pow(sys.params.B2 * k2 - sys.params.M, 2)) +
                  sys.params.M) /
                 (sys.params.A2 * kp);
    vecs(1, 3) = 0;
    vecs(2, 3) = 0;
    vecs(3, 3) = 1;

    vecs(0, 1) = 0;
    vecs(1, 1) = (sys.params.B2 * k2 +
                  std::sqrt(
                      std::pow(sys.params.A2, 2) * k2 +
                      std::pow(sys.params.B2 * k2 - sys.params.M, 2)) -
                  sys.params.M) /
                 (sys.params.A2 * kp);
    vecs(2, 1) = 1;
    vecs(3, 1) = 0;

    return vecs;
}

arma::cx_mat44 topo_orthU_3d(
    double kx, double ky, double kz, const system_data_v2& sys) {
    arma::cx_mat44 vecs{
        arma::normalise(topo_eigenvec_3d(kx, ky, kz, sys), 2, 0),
    };

    for (uint32_t i = 1; i < 4; i++) {
        for (uint32_t j = 0; j < i; j++) {
            vecs.unsafe_col(i) -=
                (arma::cdot(vecs.unsafe_col(j), vecs.unsafe_col(i)) /
                 arma::cdot(vecs.unsafe_col(j), vecs.unsafe_col(j))) *
                vecs.unsafe_col(j);
        }
        vecs.unsafe_col(i) = arma::normalise(vecs.unsafe_col(i));
    }

    return vecs;
}

arma::cx_mat44 topo_orthU_2d(double kx, double ky, const system_data_v2& sys) {
    arma::cx_mat44 vecs(arma::fill::none);

    const double k2{kx * kx + ky * ky};
    const std::complex<double> kp(kx, ky);
    const std::complex<double> km(kx, -ky);

    if (k2 < 1e-14) {
        vecs.eye();
        return vecs;
    }

    const double orth_norm_inv1{
        1.0 / (M_SQRT2 *
               std::sqrt(
                   1.0 +
                   1.0 / (std::sqrt(
                              std::pow(sys.params.A2, 2) * k2 +
                              std::pow(sys.params.M - sys.params.B2 * k2, 2)) /
                              (sys.params.B2 * k2 - sys.params.M) -
                          1.0))),
    };

    const double orth_norm_inv2{
        1.0 / (M_SQRT2 *
               std::sqrt(
                   1.0 -
                   1.0 / (std::sqrt(
                              std::pow(sys.params.A2, 2) * k2 +
                              std::pow(sys.params.M - sys.params.B2 * k2, 2)) /
                              (sys.params.B2 * k2 - sys.params.M) +
                          1.0))),
    };

    vecs(0, 0) = -(sys.params.B2 * k2 +
                   std::sqrt(
                       std::pow(sys.params.A2, 2) * k2 +
                       std::pow(sys.params.B2 * k2 - sys.params.M, 2)) -
                   sys.params.M) /
                 (sys.params.A2 * kp) * orth_norm_inv1;
    vecs(1, 0) = 0;
    vecs(2, 0) = 0;
    vecs(3, 0) = orth_norm_inv1;

    vecs(0, 2) = 0;
    vecs(1, 2) = (sys.params.B2 * k2 -
                  std::sqrt(
                      std::pow(sys.params.A2, 2) * k2 +
                      std::pow(sys.params.B2 * k2 - sys.params.M, 2)) -
                  sys.params.M) /
                 (sys.params.A2 * kp) * orth_norm_inv2;
    vecs(2, 2) = orth_norm_inv2;
    vecs(3, 2) = 0;

    vecs(0, 3) = (-sys.params.B2 * k2 +
                  std::sqrt(
                      std::pow(sys.params.A2, 2) * k2 +
                      std::pow(sys.params.B2 * k2 - sys.params.M, 2)) +
                  sys.params.M) /
                 (sys.params.A2 * kp) * orth_norm_inv2;
    vecs(1, 3) = 0;
    vecs(2, 3) = 0;
    vecs(3, 3) = orth_norm_inv2;

    vecs(0, 1) = 0;
    vecs(1, 1) = (sys.params.B2 * k2 +
                  std::sqrt(
                      std::pow(sys.params.A2, 2) * k2 +
                      std::pow(sys.params.B2 * k2 - sys.params.M, 2)) -
                  sys.params.M) /
                 (sys.params.A2 * kp) * orth_norm_inv1;
    vecs(2, 1) = orth_norm_inv1;
    vecs(3, 1) = 0;

    return vecs;
}

arma::cx_mat44 topo_vert_3d(
    const arma::vec3& Q,
    const arma::vec3& k1,
    const arma::vec3& k2,
    const system_data_v2& sys) {
    return topo_orthU_3d(
               0.5 * (Q[0] + k2[0]), 0.5 * (Q[1] + k2[1]),
               0.5 * (Q[2] + k2[2]), sys)
               .t() *
           topo_orthU_3d(
               0.5 * (Q[0] + k1[0]), 0.5 * (Q[1] + k1[1]),
               0.5 * (Q[2] + k1[2]), sys);
}

arma::cx_mat44 topo_vert_2d(
    const arma::vec2& Q, const arma::vec2& k, const system_data_v2& sys) {
    const arma::umat transf = {
        {1, 0, 0, 0},
        {0, 0, 0, 1},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
    };

    return transf *
           topo_orthU_2d(0.5 * (Q[0] + k[0]), 0.5 * (Q[1] + k[1]), sys).t() *
           topo_orthU_2d(0.5 * (Q[0] - k[0]), 0.5 * (Q[1] - k[1]), sys) *
           transf.t();
}

arma::cx_mat topo_cou_3d(
    const arma::vec3& Q,
    const arma::vec3& k1,
    const arma::vec3& k2,
    const system_data_v2& sys) {
    arma::cx_mat44 mat1{topo_vert_3d(Q, k1, k2, sys)};
    arma::cx_mat44 mat2{topo_vert_3d(Q, -k1, -k2, sys)};

    return topo_green_cou(arma::norm(k2 - k1), sys) * arma::kron(mat1, mat2);
}

arma::cx_mat topo_cou_2d(
    const arma::vec2& Q,
    const arma::vec2& k1,
    const arma::vec2& k2,
    const system_data_v2& sys) {
    arma::cx_mat44 mat1{topo_vert_2d(Q, k1, sys)};
    arma::cx_mat44 mat2{topo_vert_2d(Q, k2, sys)};

    const arma::mat transf = {
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
    };

    arma::mat eye(2, 2, arma::fill::eye);
    arma::mat cou_transf = arma::kron(eye, arma::kron(transf, eye));

    return topo_green_cou(arma::norm(k2 - k1), sys) * cou_transf *
           arma::kron(mat1, mat2) * cou_transf.t();
}

std::vector<std::complex<double>> topo_ham_3d_v(
    double kx, double ky, double kz, const system_data_v2& sys) {
    arma::cx_mat44 r(topo_ham_3d(kx, ky, kz, sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

std::vector<std::complex<double>> topo_orthU_3d_v(
    double kx, double ky, double kz, const system_data_v2& sys) {
    arma::cx_mat44 r(topo_orthU_3d(kx, ky, kz, sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

std::vector<std::complex<double>> topo_orthU_2d_v(
    double kx, double ky, const system_data_v2& sys) {
    arma::cx_mat44 r(topo_orthU_2d(kx, ky, sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

std::vector<double> topo_eigenval_3d_v(
    double k, double kz, const system_data_v2& sys) {
    arma::vec4 r(topo_eigenval_3d(k, kz, sys));

    return std::vector<double>(r.begin(), r.end());
}

std::vector<double> topo_eigenval_2d_v(double k, const system_data_v2& sys) {
    arma::vec4 r(topo_eigenval_2d(k, sys));

    return std::vector<double>(r.begin(), r.end());
}

std::vector<std::complex<double>> topo_vert_3d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys) {
    arma::cx_mat44 r(
        topo_vert_3d(arma::vec(Q), arma::vec(k1), arma::vec(k2), sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

std::vector<std::complex<double>> topo_vert_2d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys) {
    arma::cx_mat44 r(
        topo_vert_2d(arma::vec(Q), arma::vec(k1), sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

std::vector<std::complex<double>> topo_cou_3d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys) {
    arma::cx_mat r(
        topo_cou_3d(arma::vec(Q), arma::vec(k1), arma::vec(k2), sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

std::vector<std::complex<double>> topo_cou_2d_v(
    const std::vector<double>& Q,
    const std::vector<double>& k1,
    const std::vector<double>& k2,
    const system_data_v2& sys) {
    arma::cx_mat r(
        topo_cou_2d(arma::vec(Q), arma::vec(k1), arma::vec(k2), sys));

    return std::vector<std::complex<double>>(r.begin(), r.end());
}

arma::vec topo_disp_p(const arma::vec& k_vec, const system_data_v2& sys) {
    return 0.5 * std::pow(sys.c_hbarc, 2) * arma::pow(k_vec, 2) /
           sys.d_params.m_p;
}

arma::vec topo_disp_t(const arma::vec& k_vec, const system_data_v2& sys) {
    arma::vec k2_vec{arma::pow(k_vec, 2)};

    return arma::sqrt(
               4 * std::pow(sys.params.A2, 2) * k2_vec +
               arma::pow(sys.params.B2 * k2_vec - 4 * sys.params.M, 2)) -
           2 * sys.params.A2 *
               std::sqrt(
                   -std::pow(sys.params.A2, 2) +
                   4 * sys.params.B2 * sys.params.M) /
               sys.params.B2;
}

arma::vec topo_disp_t2(const arma::vec& k_vec, const system_data_v2& sys) {
    arma::vec k2_vec{arma::pow(k_vec, 2)};

    return (
        -4 * sys.params.C - k2_vec * sys.params.D2 +
        0.5 * arma::sqrt(
                  4 * std::pow(sys.params.A2, 2) * k2_vec +
                  arma::pow(sys.params.B2 * k2_vec - 4 * sys.params.M, 2)) -
        (-4 * sys.params.B2 *
             (sys.params.B2 * sys.params.C + sys.params.D2 * sys.params.M) +
         sys.params.A2 * (2 * sys.params.A2 * sys.params.D2 +
                          std::sqrt(
                              (4 * std::pow(sys.params.D2, 2) -
                               std::pow(sys.params.B2, 2)) *
                              (std::pow(sys.params.A2, 2) -
                               4 * sys.params.B2 * sys.params.M)))) /
            std::pow(sys.params.B2, 2));
}

double topo_disp_t_2d_th_f(double th, topo_disp_t_th_s* s) {
    return 0.25 * s->k *
           (4 * s->sys.params.D2 * s->k * s->Q * std::cos(th) +
            std::sqrt(
                4 * std::pow(s->sys.params.A2, 2) *
                    (s->k * s->k + s->Q * s->Q -
                     2 * s->k * s->Q * std::cos(th)) +
                std::pow(
                    4 * s->sys.params.M -
                        s->sys.params.B2 * (s->k * s->k + s->Q * s->Q) +
                        2 * s->sys.params.B2 * s->k * s->Q * std::cos(th),
                    2)) +
            std::sqrt(
                4 * std::pow(s->sys.params.A2, 2) *
                    (s->k * s->k + s->Q * s->Q +
                     2 * s->k * s->Q * std::cos(th)) +
                std::pow(
                    -4 * s->sys.params.M +
                        s->sys.params.B2 * (s->k * s->k + s->Q * s->Q) +
                        2 * s->sys.params.B2 * s->k * s->Q * std::cos(th),
                    2)));
}

struct topo_pot_cou_f {
    const system_data_v2& sys;
    double k1, k2;

    topo_pot_cou_f(const topo_pot_cou_f&) = default;
    topo_pot_cou_f(const system_data_v2& sys) : sys(sys) {}

    topo_pot_cou_f set_momentum(const double kk[2]) {
        topo_pot_cou_f new_s{topo_pot_cou_f(*this)};

        new_s.k1 = kk[0];
        new_s.k2 = kk[1];

        return new_s;
    }

    arma::vec dispersion(const arma::vec& k_vec) {
        return topo_disp_p(k_vec, sys);
    }

    double operator()(double th) {
        double k{
            std::sqrt(k1 * k1 + k2 * k2 - 2 * k1 * k2 * std::cos(th)),
        };

        if (k < 1e-5) {
            k = 1e-5;
        }

        return topo_green_cou(k, sys);
    }
};

template <typename T>
struct topo_ij_th_int_t {
    double th;

    const T& pot;

    double operator()(double th2) {
        arma::vec2 Q_vec  = {pot.Q, 0};
        arma::vec2 k1_vec = {
            pot.k1 * std::cos(th),
            pot.k1 * std::sin(th),
        };
        arma::vec2 k2_vec = {
            pot.k2 * std::cos(th2),
            pot.k2 * std::sin(th2),
        };

        if (arma::norm(k2_vec - k1_vec) < 1e-5) {
            return 0.0;
        }

        return topo_cou_2d(Q_vec, k1_vec, k2_vec, pot.sys)(pot.i, pot.j)
            .real();
    }
};

struct topo_pot_eff_cou_2d_ij_f {
    const system_data_v2& sys;
    double Q;
    uint8_t i, j;
    double k1, k2;

    topo_pot_eff_cou_2d_ij_f(const topo_pot_eff_cou_2d_ij_f&) = default;

    topo_pot_eff_cou_2d_ij_f(
        const system_data_v2& sys, double Q, uint8_t i, uint8_t j) :
        sys(sys), Q(Q), i(i), j(j) {}

    topo_pot_eff_cou_2d_ij_f(const system_data_v2& sys, uint8_t i, uint8_t j) :
        sys(sys), Q(0.0), i(i), j(j) {}

    topo_pot_eff_cou_2d_ij_f set_momentum(const double kk[2]) {
        topo_pot_eff_cou_2d_ij_f new_s{topo_pot_eff_cou_2d_ij_f(*this)};

        new_s.k1 = kk[0];
        new_s.k2 = kk[1];

        return new_s;
    }

    arma::vec dispersion(const arma::vec& k_vec) {
        arma::vec result_vec(k_vec.n_elem);

#pragma omp parallel for
        for (uint32_t i = 0; i < k_vec.n_elem; i++) {
            constexpr uint32_t n_int{1};
            constexpr uint32_t local_ws_size{(1 << 7)};
            constexpr double local_eps{1e-8};

            double result[n_int] = {0}, error[n_int] = {0};

            gsl_function integrands[n_int];

            topo_disp_t_th_s s{k_vec[i], Q, sys};

            integrands[0].function =
                &templated_f<topo_disp_t_th_s, topo_disp_t_2d_th_f>;
            integrands[0].params = &s;

            gsl_integration_workspace* ws =
                gsl_integration_workspace_alloc(local_ws_size);

            gsl_integration_qag(
                integrands, 0, 2 * M_PI, local_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result, error);

            gsl_integration_workspace_free(ws);

            result_vec[i] = result[0] * M_1_PI * 0.5;
        }

        return result_vec;
    }

    double operator()(double th) {
        constexpr uint32_t n_int{1};
        constexpr uint32_t local_ws_size{(1 << 7)};
        constexpr double local_eps{1e-8};

        double result[n_int] = {0}, error[n_int] = {0};

        gsl_function integrands[n_int];

        topo_ij_th_int_t<topo_pot_eff_cou_2d_ij_f> s{th, *this};

        integrands[0].function =
            &functor_call<topo_ij_th_int_t<topo_pot_eff_cou_2d_ij_f>>;
        integrands[0].params = &s;

        gsl_integration_workspace* ws =
            gsl_integration_workspace_alloc(local_ws_size);

        gsl_integration_qag(
            integrands, 0, 2 * M_PI, local_eps, 0, local_ws_size,
            GSL_INTEG_GAUSS31, ws, result, error);

        gsl_integration_workspace_free(ws);

        return result[0] * 0.5 * M_1_PI;
    }
};

template <typename T>
struct topo_th_int_t {
    double th;

    const T& pot;

    double operator()(double th2) {
        arma::vec2 Q_vec  = {pot.Q, 0};
        arma::vec2 k1_vec = {
            pot.k1 * std::cos(th),
            pot.k1 * std::sin(th),
        };
        arma::vec2 k2_vec = {
            pot.k2 * std::cos(th2),
            pot.k2 * std::sin(th2),
        };

        if (arma::norm(k2_vec - k1_vec) < 1e-5) {
            return 0.0;
        }

        arma::mat22 result{
            arma::real(topo_cou_2d(Q_vec, k1_vec, k2_vec, pot.sys)
                           .submat(1, 1, 2, 2)),
        };

        return 2 * (result(1, 1) - pot.alpha * result(0, 1));
    }
};

struct topo_pot_eff_cou_2d_f {
    const system_data_v2& sys;
    double Q;
    double alpha;
    double k1, k2;

    topo_pot_eff_cou_2d_f(const topo_pot_eff_cou_2d_f&) = default;

    topo_pot_eff_cou_2d_f(const system_data_v2& sys, double Q, double alpha) :
        sys(sys), Q(Q), alpha(alpha) {}

    topo_pot_eff_cou_2d_f(const system_data_v2& sys, double alpha) :
        sys(sys), Q(0.0), alpha(alpha) {}

    topo_pot_eff_cou_2d_f set_momentum(const double kk[2]) {
        topo_pot_eff_cou_2d_f new_s{topo_pot_eff_cou_2d_f(*this)};

        new_s.k1 = kk[0];
        new_s.k2 = kk[1];

        return new_s;
    }

    arma::vec dispersion(const arma::vec& k_vec) {
        arma::vec result_vec(k_vec.n_elem);

#pragma omp parallel for
        for (uint32_t i = 0; i < k_vec.n_elem; i++) {
            constexpr uint32_t n_int{1};
            constexpr uint32_t local_ws_size{(1 << 7)};
            constexpr double local_eps{1e-8};

            double result[n_int] = {0}, error[n_int] = {0};

            gsl_function integrands[n_int];

            topo_disp_t_th_s s{k_vec[i], Q, sys};

            integrands[0].function =
                &templated_f<topo_disp_t_th_s, topo_disp_t_2d_th_f>;
            integrands[0].params = &s;

            gsl_integration_workspace* ws =
                gsl_integration_workspace_alloc(local_ws_size);

            gsl_integration_qag(
                integrands, 0, 2 * M_PI, local_eps, 0, local_ws_size,
                GSL_INTEG_GAUSS31, ws, result, error);

            gsl_integration_workspace_free(ws);

            result_vec[i] = result[0] * M_1_PI * 0.5;
        }

        return result_vec;
    }

    double operator()(double th) {
        constexpr uint32_t n_int{1};
        constexpr uint32_t local_ws_size{(1 << 7)};
        constexpr double local_eps{1e-8};

        double result[n_int] = {0}, error[n_int] = {0};

        gsl_function integrands[n_int];

        topo_th_int_t<topo_pot_eff_cou_2d_f> s{th, *this};

        integrands[0].function =
            &functor_call<topo_th_int_t<topo_pot_eff_cou_2d_f>>;
        integrands[0].params = &s;

        gsl_integration_workspace* ws =
            gsl_integration_workspace_alloc(local_ws_size);

        gsl_integration_qag(
            integrands, 0, 2 * M_PI, local_eps, 0, local_ws_size,
            GSL_INTEG_GAUSS31, ws, result, error);

        gsl_integration_workspace_free(ws);

        return result[0] * 0.5 * M_1_PI;
    }
};

template <
    typename topo_functor,
    bool force_positive = true,
    bool only_sign      = false>
double topo_det_f(double z, topo_mat_s<topo_functor>* s) {
    if constexpr (force_positive) {
        z = std::exp(z);
    }

    s->update_mat_potcoef(z);

    double val, sign, valint;
    arma::log_det(val, sign, s->mat_potcoef);

    if constexpr (only_sign) {
        return sign;
    } else {
        return sign * std::exp(val + std::log(1e20));
    }
}

template <typename topo_functor, bool use_brackets = false>
double topo_det_zero_t(topo_mat_s<topo_functor>& s, double be_bnd) {
    /*
     * be_bnd: upper bound for the binding energy -> positive value!
     */

    constexpr double local_eps{1e-8}, be_min{0.2};
    double z, z_min{std::log(be_min)}, z_max{std::log(std::abs(be_bnd))},
        z0{std::log(be_bnd)};
    // double z, z_min{be_min}, z_max{std::abs(be_bnd)}, z0{be_bnd};

    s.fill_mat_potcoef();
    s.fill_mat_elem();

    if constexpr (use_brackets) {
        gsl_function funct;
        /*
        funct.function = &median_f<
            void,
            templated_f<
                topo_mat_s<topo_functor, dispersion_func>,
                topo_det_f<topo_functor, dispersion_func, true, true>>,
            5>;
        */
        funct.function = &templated_f<
            topo_mat_s<topo_functor>, topo_det_f<topo_functor, true, true>>;
        funct.params = &s;

        double f_min{funct.function(z_min, funct.params)};
        double f_max{funct.function(z_max, funct.params)};

        ///*
        printf(
            "[%s] f: (%e, %e), z: (%f, %f)\n", __func__, f_min, f_max, z_min,
            z_max);
        //*/

        if (f_min * f_max > 0) {
            printf(
                "\033[91;1m[%s] Ends of the bracket must have opposite "
                "signs.\033[0m\n",
                __func__);
            for (uint32_t i = 0; i < max_iter; i++) {
                z_min = std::log(be_min * std::pow(1.05, i + 1));
                f_min = funct.function(z_min, funct.params);

                if (f_min * f_max < 0) {
                    break;
                } else if (i + 1 == max_iter) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
            }
        }

        // const gsl_root_fsolver_type* solver_type =
        // gsl_root_fsolver_brent;
        const gsl_root_fsolver_type* solver_type = gsl_root_fsolver_bisection;
        gsl_root_fsolver* solver = gsl_root_fsolver_alloc(solver_type);

        gsl_root_fsolver_set(solver, &funct, z_min, z_max);

        for (int status = GSL_CONTINUE, iter = 0;
             status == GSL_CONTINUE && iter < 16; iter++) {
            status = gsl_root_fsolver_iterate(solver);
            z      = gsl_root_fsolver_root(solver);
            z_min  = gsl_root_fsolver_x_lower(solver);
            z_max  = gsl_root_fsolver_x_upper(solver);

            ///*
            printf("[%s]: <%d> z: (%e, %e)\n", __func__, iter, z_min, z_max);
            //*/

            /*
            printf(
                "[%s]: <%d> f: (%e, %e), z: (%f, %f)\n", __func__, iter,
                funct.function(z_min, funct.params),
                funct.function(z_max, funct.params), z_min, z_max);
            */

            status = gsl_root_test_interval(z_min, z_max, 0, local_eps);
        }

        gsl_root_fsolver_free(solver);

        z = std::exp(z);
    } else {
        gsl_function_fdf funct;
        funct.f      = &templated_f<topo_mat_s<topo_functor>, topo_det_f>;
        funct.df     = &templated_df<topo_mat_s<topo_functor>, topo_det_f>;
        funct.fdf    = &templated_fdf<topo_mat_s<topo_functor>, topo_det_f>;
        funct.params = &s;

        if (std::abs(funct.f(z0, funct.params)) < local_eps) {
            return std::exp(z0);
        }

        const gsl_root_fdfsolver_type* solver_type =
            gsl_root_fdfsolver_steffenson;
        gsl_root_fdfsolver* solver = gsl_root_fdfsolver_alloc(solver_type);

        gsl_root_fdfsolver_set(solver, &funct, z0);

        for (int status = GSL_CONTINUE, iter = 0;
             status == GSL_CONTINUE && iter < max_iter; iter++) {
            z0     = z;
            status = gsl_root_fdfsolver_iterate(solver);
            z      = gsl_root_fdfsolver_root(solver);

            double fval{funct.f(z, funct.params)};

            status = gsl_root_test_residual(fval, local_eps);

            /*
            printf("[%s]: <%d> f: (%e), z: (%f)\n", __func__, iter, fval,
            z);
            */
        }

        gsl_root_fdfsolver_free(solver);
        z = std::exp(z);
    }

    return z;
}

std::vector<double> topo_det_p_cou_vec(
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys) {
    using pot_functor       = topo_pot_cou_f;
    constexpr auto det_func = topo_det_f<pot_functor, false>;

    pot_functor pot_s(sys);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    mat_s.fill_mat_potcoef();
    mat_s.fill_mat_elem();

    std::vector<double> r(z_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < z_vec.size(); i++) {
        r[i] = det_func(z_vec[i], &mat_s);
    }

    return r;
}

std::vector<double> topo_det_t_eff_cou_vec(
    double alpha,
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys) {
    using pot_functor       = topo_pot_eff_cou_2d_f;
    constexpr auto det_func = topo_det_f<pot_functor, false>;

    pot_functor pot_s(sys, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    mat_s.fill_mat_potcoef();
    mat_s.fill_mat_elem();

    std::vector<double> r(z_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < z_vec.size(); i++) {
        r[i] = det_func(z_vec[i], &mat_s);
        /*
        double int_part;
        r[i] = (z_vec[i] > 0 ? 1.0 : -1.0) *
               std::log(1 + std::exp(std::modf(z_vec[i], &int_part)));
        */
    }

    return r;
}

std::vector<double> topo_det_t_eff_cou_Q_vec(
    double Q,
    double alpha,
    const std::vector<double>& z_vec,
    uint32_t N_k,
    const system_data_v2& sys) {
    using pot_functor       = topo_pot_eff_cou_2d_f;
    constexpr auto det_func = topo_det_f<pot_functor, false>;

    pot_functor pot_s(sys, Q, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    mat_s.fill_mat_potcoef();
    mat_s.fill_mat_elem();

    std::vector<double> r(z_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < z_vec.size(); i++) {
        r[i] = det_func(z_vec[i], &mat_s);
    }

    return r;
}

double topo_be_p_cou(uint32_t N_k, const system_data_v2& sys, double be_bnd) {
    using pot_functor       = topo_pot_cou_f;
    constexpr auto det_zero = topo_det_zero_t<pot_functor, true>;

    pot_functor pot_s(sys);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    return det_zero(mat_s, be_bnd);
}

double topo_be_t_eff_cou(
    double alpha, uint32_t N_k, const system_data_v2& sys, double be_bnd) {
    using pot_functor       = topo_pot_eff_cou_2d_f;
    constexpr auto det_zero = topo_det_zero_t<pot_functor>;

    pot_functor pot_s(sys, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    return det_zero(mat_s, be_bnd);
}

double topo_be_t_eff_cou_Q(
    double Q,
    double alpha,
    uint32_t N_k,
    const system_data_v2& sys,
    double be_bnd) {
    using pot_functor       = topo_pot_eff_cou_2d_f;
    constexpr auto det_zero = topo_det_zero_t<pot_functor>;

    pot_functor pot_s(sys, Q, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    return det_zero(mat_s, be_bnd);
}

double topo_be_b_t_eff_cou_Q(
    double Q,
    double alpha,
    uint32_t N_k,
    const system_data_v2& sys,
    double be_bnd) {
    using pot_functor       = topo_pot_eff_cou_2d_f;
    constexpr auto det_zero = topo_det_zero_t<pot_functor, true>;

    pot_functor pot_s(sys, Q, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);

    return det_zero(mat_s, be_bnd);
}

std::vector<double> topo_cou_mat(uint32_t N_k, const system_data_v2& sys) {
    using pot_functor = topo_pot_cou_f;

    pot_functor pot_s(sys);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);
    mat_s.fill_mat_potcoef();

    return std::vector<double>(
        mat_s.mat_potcoef.begin(), mat_s.mat_potcoef.end());
}

std::vector<double> topo_eff_cou_ij_mat(
    uint8_t mat_i, uint8_t mat_j, uint32_t N_k, const system_data_v2& sys) {
    using pot_functor = topo_pot_eff_cou_2d_ij_f;

    pot_functor pot_s(sys, mat_i, mat_j);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);
    mat_s.fill_mat_potcoef();

    return std::vector<double>(
        mat_s.mat_potcoef.begin(), mat_s.mat_potcoef.end());
}

std::vector<double> topo_eff_cou_mat(
    double alpha, uint32_t N_k, const system_data_v2& sys) {
    using pot_functor = topo_pot_eff_cou_2d_f;

    pot_functor pot_s(sys, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);
    mat_s.fill_mat_potcoef();

    return std::vector<double>(
        mat_s.mat_potcoef.begin(), mat_s.mat_potcoef.end());
}

std::vector<double> topo_eff_cou_Q_ij_mat(
    double Q,
    uint8_t mat_i,
    uint8_t mat_j,
    uint32_t N_k,
    const system_data_v2& sys) {
    using pot_functor = topo_pot_eff_cou_2d_ij_f;

    pot_functor pot_s(sys, Q, mat_i, mat_j);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);
    mat_s.fill_mat_potcoef();

    return std::vector<double>(
        mat_s.mat_potcoef.begin(), mat_s.mat_potcoef.end());
}

std::vector<double> topo_eff_cou_Q_mat(
    double Q, double alpha, uint32_t N_k, const system_data_v2& sys) {
    using pot_functor = topo_pot_eff_cou_2d_f;

    pot_functor pot_s(sys, Q, alpha);
    topo_mat_s<pot_functor> mat_s(N_k, pot_s);
    mat_s.fill_mat_potcoef();

    return std::vector<double>(
        mat_s.mat_potcoef.begin(), mat_s.mat_potcoef.end());
}

std::vector<double> topo_be_t_eff_cou_vec(
    const std::vector<double>& alpha_vec,
    uint32_t N_k,
    const system_data_v2& sys,
    double be_bnd) {
    using pot_functor       = topo_pot_eff_cou_2d_f;
    constexpr auto det_zero = topo_det_zero_t<pot_functor>;

    std::vector<double> r(alpha_vec.size());

#pragma omp parallel for
    for (uint32_t i = 0; i < alpha_vec.size(); i++) {
        pot_functor pot_s(sys, alpha_vec[i]);
        topo_mat_s<pot_functor> mat_s(N_k, pot_s);

        r[i] = det_zero(mat_s, be_bnd);
    }

    return r;
}
