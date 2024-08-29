#import
from setuptools import setup, find_packages
from typing import List

#read requirements.txt
HYPHEN_E_DOT = '-e .'
def get_requirements(file_path:str) ->List[str]:
    requirements = []
    with open(file_path) as obj:
         requirements= obj.readlines()
         requirements = [o.replace("\n","") for o in requirements]
         if HYPHEN_E_DOT in requirements:
             requirements.remove(HYPHEN_E_DOT)
    return requirements

# setup()
setup(
    name="MachineLearningProject",
    version="0.0.1",
    author="Emery Li",
    author_email="71569536+Emeryli@users.noreply.github.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)
install_requires= get_requirements('requirements.txt') 
print(install_requires)
