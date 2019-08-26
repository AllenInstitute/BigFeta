#!/usr/bin/env python
from setuptools import setup,find_packages
import sys
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import shlex
        import pytest
        self.pytest_args += " --cov=bigfeta --cov-report html "\
                            "--junitxml=test-reports/test.xml"

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(name='bigfeta',
      use_scm_version=True,
      description='a python package to solve for transformations of image tiles, given point matches' 
                  'between those tiles and their transformations stored in a render database '
                  'databases see https://github.com/saalfeldlab/render and https://github.com/khaledkhairy/EM_aligner',
      author='Daniel Kapner',
      author_email='danielk@alleninstitute.org',
      url='https://github.com/AllenInstitute/bigfeta',
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required,
      tests_require=test_required,
      cmdclass={'test': PyTest})
