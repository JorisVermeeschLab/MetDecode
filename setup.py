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
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'torch',
        'tqdm',
    ],
    extras_require={
        'analysis': [
            'matplotlib',
            'scikit-learn',
            'seaborn',
        ],
    },
    entry_points={
        'console_scripts': [
            'metdecode=metdecode.cli:main',
        ],
    },
    include_package_data=False
)
