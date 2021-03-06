#!/usr/bin/env python3
import os
import sys


def find_files(path, ext):
    files = []

    for entry in os.listdir(path):
        if os.path.splitext(entry)[1] == ext:
            files.append('%s/%s' % (path, entry))

    return files


src_path = 'src'
swig_path = 'swig'

includes = ['include', '/usr/include/python3.9', '/usr/local/include']

output_bin = 'bin'
output_lib = output_bin + '/semiconductor.so'
output_swig_so = 'src/semiconductor_wrapper.os'
output_py = 'bin/main.py'
output_main = output_bin + '/main'

lib_sources = [e for e in find_files(src_path, '.cpp') if not 'wrapper' in e
               ] + find_files(src_path, '.cc')

swig_sources = find_files(swig_path, '.i')
py_modules = find_files('python', '.py')

swig_py = 'bin/semiconductor.py'
swig_cpp = 'src/semiconductor_wrapper.cpp'

cc_flags = [
    '-O3',
    '-Wall',
    '-std=c++17',
    '-pedantic',
    '-march=native',
    '-fopenmp',
    '-Wno-unused-variable',
    '-Wno-deprecated-declarations',
]
incl_libs = [
    'gsl',
    'cblas',
    'm',
    'gmpxx',
    'mpfr',
    'gmp',
    'arb',
    'itpp',
    'armadillo',
    'gomp',
]
swig_flags = [
    '-python',
    '-builtin',
    '-py3',
    '-threads',
    '-c++',
    '-fcompact',
    '-Wall',
    '-fastdispatch',
    '-fvirtual',
    '-fastproxy',
    '-dirvtable',
]

swig_cmd = 'swig %s -o $TARGET -Iinclude -outdir %s $SOURCE' % (
    ' '.join(swig_flags), output_bin)

cpp_compiler = 'clang++'

env = Environment(CXX=cpp_compiler)

env['ENV']['TERM'] = os.environ['TERM']

env.Alias('lib', output_lib)

env.Append(CPPPATH=includes)
env.Replace(CCFLAGS=cc_flags)
env.Append(LIBS=incl_libs)
env.Append(LINKFLAGS=['-fopenmp'])

env.Append(SWIGFLAGS=swig_flags)
env.Replace(SWIGPATH=includes[0])
env.Replace(SWIGOUTDIR=output_bin)
env.Replace(SWIGCXXFILESUFFIX='cpp')

env.ParseConfig('python-config --libs')

py = env.Install(output_bin, py_modules)
swig_gen = env.Command(swig_cpp, swig_sources, swig_cmd)
swig_so = env.SharedObject(
    target=output_swig_so,
    source=swig_cpp,
)
main = env.Program(target=output_main,
                   source=lib_sources,
                   LIBPATH=['/usr/lib', '/usr/local/lib'])
lib = env.SharedLibrary(target=output_lib,
                        source=[output_swig_so] + lib_sources,
                        SHLIBPREFIX='_',
                        LIBPATH=['/usr/lib', '/usr/local/lib'])

Clean(py, ['bin/__pycache__'])
Clean(swig_gen, swig_py)

Requires(lib, main)
Requires(lib, py)

Default(lib)
