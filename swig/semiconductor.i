#define SWIGPYTHON_BUILTIN

%module semiconductor

%include "stdint.i"
%include "std_string.i"
%include "std_complex.i"
%include "std_vector.i"
%include "std_array.i"
%include "std_map.i"

namespace std {
    %template(DoubleVector) vector<double>;
    %template(Uint32Vector) vector<uint32_t>;
    %template(ComplexDoubleVector) vector<std::complex<double>>;
    %template(DoubleVectorVector) vector<vector<double>>;
    %template(Uint32VectorVector) vector<vector<uint32_t>>;
    %template(MapStringDouble) map<string, double>;
}

%{
#include "common.h"
#include "utils.h"
#include "wavefunction.h"
#include "wavefunction_bexc.h"
#include "analytic.h"
#include "fluctuations.h"
#include "analytic_2d.h"
#include "fluctuations_2d.h"
#include "plasmons.h"
#include "biexcitons.h"
#include "excitons.h"
#include "Faddeeva.hh"
#include "topo.h"
%}


%include "common.h"
%include "utils.h"
%include "wavefunction.h"
%include "wavefunction_bexc.h"
%include "analytic.h"
%include "fluctuations.h"
%include "analytic_2d.h"
%include "fluctuations_2d.h"
%include "plasmons.h"
%include "biexcitons.h"
%include "excitons.h"
%include "Faddeeva.hh"
%include "topo.h"

%template(Result1) result_s<1>;
%template(Result2) result_s<2>;
%template(Result7) result_s<7>;
%template(VectorResult1) std::vector<result_s<1>>;
%template(VectorResult2) std::vector<result_s<2>>;
%template(VectorResult7) std::vector<result_s<7>>;
