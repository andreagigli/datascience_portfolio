from setuptools import setup, find_packages
from os import path

# Read the contents of your requirements.txt file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = f.read().splitlines()

setup(
    name='portfolio_ML_datascience',
    description='Custom package for ML and data science',
    version='0.1.0',
    packages=find_packages(),  # Automatically finds and lists all packages in your project
    install_requires=install_requires,  # Gets the list of requirements from the requirements.txt file
    python_requires='>=3.10',  # Your project's Python version requirement
    author='Andrea Gigli',
    author_email='gigli.andrea91@gmail.com',
    url='URL to your project repository',
    license='GNU General Public License (GPL)',
    keywords='Some relevant keywords for your project'
)
