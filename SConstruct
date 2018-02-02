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

includes = [ 'include', '/usr/include/python3.6m' ]

output_bin = 'bin'
output_main = output_bin + '/ctemp'
output_lib = output_bin + '/integrals.so'
output_swig_so = 'src/integrals_wrapper.os'

cpp_sources = '%s/%s' % (src_path, 'main.cpp')
lib_sources = [ e for e in find_files(src_path, '.cpp') if not 'main' in e and not 'wrapper' in e ]
swig_sources = find_files(swig_path, '.i')

swig_py = 'bin/integrals.py'
swig_cpp = 'src/integrals_wrapper.cpp'

cc_flags = ['-O3', '-Wall', '-std=c++17', '-pedantic']
incl_libs = ['gsl', 'cblas', 'm', 'gmpxx', 'mpfr', 'gmp', 'arb']
swig_flags = ['-python', '-builtin', '-py3', '-threads', '-O', '-c++', '-fcompact', '-Wall']

swig_cmd = 'swig %s -o $TARGET -Iinclude -outdir %s $SOURCE' % (' '.join(swig_flags), output_bin)

cpp_compiler = 'clang++'

env = Environment(CXX = cpp_compiler)

env['ENV']['TERM'] = os.environ['TERM']

env.Alias('lib', output_lib)
env.Alias('main', output_main)

env.Append(CPPPATH = includes)
env.Replace(CCFLAGS = cc_flags)
env.Append(LIBS = incl_libs)

env.Append(SWIGFLAGS = swig_flags)
env.Replace(SWIGPATH = includes[0])
env.Replace(SWIGOUTDIR = output_bin)
env.Replace(SWIGCXXFILESUFFIX = 'cpp')

env.ParseConfig("python-config --libs")

py = env.InstallAs('bin/main.py', 'python/c_api_test.py')
swig_gen = env.Command(swig_cpp, swig_sources, swig_cmd)
swig_so = env.SharedObject(target = output_swig_so, source = swig_cpp)
lib = env.SharedLibrary(target = output_lib, source = [ output_swig_so ] + lib_sources, SHLIBPREFIX = '_')
main = env.Program(target = output_main, source = cpp_sources, LIBS = incl_libs + [lib])

Clean(py, 'bin/__pycache__')
Clean(swig_gen, swig_py)

Requires(lib, py)
Requires(main, lib)

Default(main)

