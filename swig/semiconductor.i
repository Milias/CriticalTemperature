#define SWIGPYTHON_BUILTIN

%module semiconductor

%include "stdint.i"
%include "std_complex.i"
%include "std_vector.i"
%include "std_array.i"

%template(DoubleVector) std::vector<double>;
%template(Uint32Vector) std::vector<uint32_t>;
%template(ComplexDoubleVector) std::vector<std::complex<double>>;
%template(DoubleVectorVector) std::vector<std::vector<double>>;
%template(Uint32VectorVector) std::vector<std::vector<uint32_t>>;

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
#include "Faddeeva.hh"
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
%include "Faddeeva.hh"

%template(Result1) result_s<1>;
%template(Result2) result_s<2>;
%template(Result7) result_s<7>;
%template(VectorResult1) std::vector<result_s<1>>;
%template(VectorResult2) std::vector<result_s<2>>;
%template(VectorResult7) std::vector<result_s<7>>;
