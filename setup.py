from pathlib import Path
from setuptools import setup


long_description = (
    Path(__file__).parent / "README.md"
).read_text(encoding="utf-8")

packages = [
    'metdecode',
]

setup(
    name='metdecode',
    version='1.0.1',
    description='Reference-based Deconvolution of Whole-Genome Methylation Sequencing data',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
