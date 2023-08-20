from setuptools import setup, find_packages

setup(
    name='Online Panel Method Calculator',
    version='0.1',
    author='Roman Lemieszynski',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask', "numpy", "matplotlib"
    ],
)
