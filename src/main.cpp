#include "biexcitons.h"

int main(int argc, char** argv) {
    initializeMPFR_GSL();
    double m_e = 0.22, m_h = 0.41, eps_r = 6.369171898453055, T = 294;
    system_data sys = system_data(m_e, m_h, eps_r, T);

    uint32_t N_param{1 << 10};
    double x0{1e-1}, xf{1e1};
    std::vector<double> result_vec(N_param);

    return 0;
}

