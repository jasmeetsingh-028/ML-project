## making machine learing application as a package

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str)->List[str]:
    '''
    this function returns the list of requirements for the project.
    '''

    requirements =[]
    with open(file_path) as file_obj:

        requirements = file_obj.readlines()
        #\n will be added after reading lines
        #replace it with blank

        requirements = [req.replace ("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    
    return requirements


setup(
name = 'MyMLproject',
version = '0.0.1',
author = 'Jasmeet Singh',
author_email = 'sjasmeet135@gmail.com',
packages= find_packages(),
install_requires = get_requirements('requirements.txt')

)