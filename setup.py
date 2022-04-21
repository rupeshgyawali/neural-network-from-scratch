from setuptools import setup, find_packages

setup(
    name='neural-network',
    version='0.0.1',
    description='A library for neural network models built from scratch',
    author='Rupesh Gyawali',
    url='https://github.com/rupeshgyawali/neural-network-from-scratch',
    packages=find_packages(include=['neural_network']),
    install_requires=['numpy==1.22.3'],
)

