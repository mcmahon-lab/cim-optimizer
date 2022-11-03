from setuptools import setup, find_packages
import os

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='cim-optimizer',
      version=f'1.0.3',
      description='Simulated Implementation of the Coherent Ising Machine',
      author='McMahon Lab',
      author_email='pmcmahon@cornell.edu',
      license='CC-BY-4.0 license',
      license_files = ('LICENSE.txt',),
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/mcmahon-lab/cim-optimizer',
      packages=find_packages(),
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
      ],
      python_requires='>=3.7',
      install_requires=[
          'numpy',
          'BOHB_HPO',
          'matplotlib',
          'torch']
)
