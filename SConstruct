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
header_path = 'include'
output_target_main = 'bin/ctemp'
output_target_lib = 'bin/libintegrals.so'

cpp_sources = '%s/%s' % (src_path, 'main.cpp')
lib_sources = [ e for e in find_files(src_path, '.cpp') if not 'main' in e ]

cc_flags = ['-O3', '-Wall', '-std=c++17', '-pedantic']
incl_libs = ['gsl', 'cblas', 'm', 'gmpxx', 'mpfr', 'gmp', 'arb']

cpp_compiler = 'clang++'

env = Environment(CXX = cpp_compiler)

env['ENV']['TERM'] = os.environ['TERM']

env.Alias('lib', output_target_lib)
env.Alias('main', output_target_main)

env.Append(CPPPATH = [ header_path ])
env.Replace(CCFLAGS = cc_flags)
env.Append(LIBS = incl_libs)

lib = env.SharedLibrary(target = output_target_lib, source = lib_sources)
main = env.Program(target = output_target_main, source = cpp_sources, LIBS = incl_libs + [ lib ])

Default(main)

