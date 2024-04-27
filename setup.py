from setuptools import setup, find_packages

setup(
    name='allclear',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'rasterio',
        'numpy',
        'matplotlib',
        'pandas',
    ],
    url='https://github.com/Zhou-Hangyu/allclear',
    license='MIT',
    description='A comprehensive benchmark for cloud removal.',
)