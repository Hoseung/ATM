from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding='utf-8')

setup(
    name='atm',
    version='0.0.1',
    packages=find_packages(),
    description='Tone mapping astronomical images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Hoseung/ATM',
    classifiers=['Programming Language :: Python :: 3'],
    python_requires='>=3.8',
    install_requires=['numpy', 'scikit-image', 'astropy', 'sklearn', 'matplotlib',
                      'colour-science', 'ax-platform', 'photutils', 'tqdm', 
                      'tensorflow', 'pyyaml'],
    extras_require={'dev':[''],
                    'test':['']},
)
