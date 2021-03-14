import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.4.0'
PACKAGE_NAME = 'CoNSEPT'
AUTHOR = 'Payam Dibaeinia'
AUTHOR_EMAIL = 'dibaein2@illinois.edu'
URL = 'https://github.com/PayamDiba/CoNSEPT'

LICENSE = 'MIT License'
DESCRIPTION = 'A Convolutional Neural Network-based Sequence-to-Expression Prediction Tool'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'tensorflow>=2.3',
      'numpy',
      'pandas',
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages(),
      entry_points ={'console_scripts': [ 'consept = CoNSEPT.train:main']},
      python_requires='>3.5.2',
      )
