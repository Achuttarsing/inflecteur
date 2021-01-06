# setup.py
from setuptools import setup

setup(
  name="inflecteur",
  version="0.1.1",
  packages=["inflecteur"],
  license="MIT",
  author="Adrien Chuttarsing",
  author_email="adrien.chuttarsing@gmail.com",
  url="https://github.com/Achuttarsing/inflecteur",
  description="python inflector for French language : control gender, tense and number",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  keywords="nlp inflector python inflecteur data augmentation french",
  install_requires = ['sacremoses','tokenizers','sentencepiece','transformers','pandas','numpy'],
  classifiers=[
    "Natural Language :: French",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic"
  ],
)
