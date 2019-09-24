#include "biexcitons.h"

int main(int argc, char** argv) {
    initializeMPFR_GSL();
    double m_e = 0.22, m_h = 0.41, eps_r = 6.369171898453055, T = 294;
    system_data sys = system_data(m_e, m_h, eps_r, T);

    uint32_t N_param{1 << 10};
    double x0{1e-1}, xf{1e1};
    std::vector<double> result_vec(N_param);

#pragma omp parallel for
    for (uint32_t i = 0; i < N_param; i++) {
        double x{x0 + i * (xf - x0) / N_param};
        COZ_BEGIN("Delta");
        result_vec[i] = biexciton_Delta_r(x, sys).value[0];
        COZ_END("Delta");

        COZ_BEGIN("J");
        result_vec[i] = biexciton_J_r(x, sys).value[0];
        COZ_END("J");

        COZ_BEGIN("Jp");
        result_vec[i] = biexciton_Jp_r(x, sys).value[0];
        COZ_END("Jp");

        COZ_BEGIN("K");
        result_vec[i] = biexciton_K_r(x, sys).value[0];
        COZ_END("K");
    }

    return 0;
}

