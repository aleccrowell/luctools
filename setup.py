from setuptools import setup

setup(name='luctools',
      version='0.2',
      packages=['luctools'],
      py_modules=['analyse'],
      description='Detrending, period/phase prediction and plotting of luciferase time series',
      url='https://github.com/aleccrowell/luctools',
      download_url='https://github.com/aleccrowell/luctools/releases/tag/v0.2',
      author='Alec Crowell',
      author_email='alexander.m.crowell@gmail.com',
      license='BSD-3',
      keywords=['circadian','luciferase'],
      install_requires=['numpy','pandas','scipy','matplotlib','seaborn','peakutils'],
      zip_safe=False,long_description=open('README.md').read())
