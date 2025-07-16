from setuptools import setup, find_packages
from typing import List
HYPEN_E_DOT = '-e .' # This will help requirement files to be installed with setup.py file

def get_requirements(file_path:str)->List[str]:
    """
    Reads a requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(name='end_mlprojects',
      version='0.1',
      description='End-to-end machine learning projects',
      author='Vishesh Saxena',
      packages= find_packages(),
      install_requires=get_requirements('requirements.txt'))