import glob
import regions
from distutils.core import setup

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

atlases = [f.split('regions/')[1] for f in glob.glob('regions/atlases/*.nii.gz')]
labels =  [f.split('regions/')[1] for f in glob.glob('regions/atlases/*.xml')]

print atlases, labels

setup(
    name='regions',
    description='A module for manipulating regions of interest in images',
    # long_description=open('README.rst').read(),
    version=regions.__version__,
    author='Yannick Schwartz',
    packages = ['regions'],
    package_data={'regions': atlases + labels},
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
)
