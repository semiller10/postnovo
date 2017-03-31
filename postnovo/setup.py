from setuptools import setup, find_packages

setup(name = 'postnovo',
      version = 0.1,
      description = 'postnovo post-processes de novo sequences to improve their accuracy',
      author = 'Samuel Miller',
      license = 'Apache',
      packages = ['postnovo'],
      entry_points = {'console_scripts': ['postnovo = postnovo.postnovo:main']},
      install_requires = ['numpy >= 1.11.0', 'pandas >= 0.19.2', 'scikit-learn >= 0.18.1'],
      classifiers = [
          'Development Status :: 4 - Beta',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5'
          ],
      keywords = 'proteomics LC-MS/MS MS/MS',
      url = 'https://github.com/semiller10/postnovo'
)