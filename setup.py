from setuptools import setup


packages = [
    'metdecode',
]

setup(
    name='metdecode',
    version='1.0.0',
    description='Reference-based Deconvolution of Whole-Genome Methylation Sequencing data',
    url='https://github.com/JorisVermeeschLab/MetDecode',
    author='Antoine Passemiers',
    packages=packages,
    include_package_data=False
)
