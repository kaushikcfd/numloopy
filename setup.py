#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ver_dic = {}
version_file = open("faster_array/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

exec(compile(version_file_contents, "faster_array/version.py", 'exec'), ver_dic)

setup(name="faster_array",
      version="2017.1",
      description="Lazy evaluation for array expressions, based on loopy",
      long_description=open("README.rst").read(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
          ],

      install_requires=[
          "loo.py",
          "numpy>=1.6.0",
          ],

      author="Andreas Kleockner",
      # url="http://pypi.python.org/pypi/pytools",
      author_email="svlin2@illinois.edu",
      license="MIT",
      packages=find_packages())
