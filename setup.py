from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with (this_directory / "linac" / "version.py").open() as f:
    version = f.read().split(" = '")[1].split("'\n")[0]


setup(
    name='linac',
    version=version,
    license='GNU General Public License v3.0',
    description='Linear Algebra with CUDA',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy<2.0',
        'mpmath',
        'pyadic',
        'diskcache',
    ],
    extras_require={
        'full': [
            'pycuda',
        ]
    }
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
