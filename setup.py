from setuptools import setup, find_packages


setup(
    name='linac',
    version='0.0.1',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    description='Linear Algebra with CUDA',
    packages=find_packages(),
    include_package_data=True,
    # data_files=[('linac', [..., ...])],
    install_requires=['numpy', ],
)
