from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='retrofit-cost-tool',
    version='1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    author='Juan F. Fung',
    author_email='juan.fung@nist.gov',
    description='A package for predicting seismic retrofit costs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://sgitlab.nist.gov/jff/retrofit-cost-tool.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12,<3.13',
)
