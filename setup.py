# setup.py
from setuptools import setup

setup(
  name="inflecteur",
  version="0.1",
  packages=["inflecteur"],
  license="MIT",
  author="Adrien Chuttarsing",
  author_email="adrien.chuttarsing@gmail.com",
  url="https://github.com/Achuttarsing/inflecteur",
  description="python inflector for French language : control gender, tense and number",
  long_description=open("README.md").read(),
  long_description_content_type="text/markdown",
  keywords="nlp inflector python inflecteur data augmentation french",
  install_requires = ['sentencepiece','transformers','pandas','numpy','glob','zipfile'],
  classifiers=[
      "Programming Language :: Python :: 3",
      "Topic :: Text Processing :: Linguistic",
      "Natural Language :: French"
  ],
)
