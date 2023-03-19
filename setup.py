from distutils.core import setup

from setuptools import find_packages

import HAI_NN


def get_readme():
    with open('README.md', encoding='utf-8') as readme_file:
        return readme_file.read()


setup(
    name='alive-progress',
    version=HAI_NN.__version__,
    description=HAI_NN.__description__,
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/HaithamIsmail/HAI_NN',
    author=HAI_NN.__author__,
    author_email=HAI_NN.__email__,
    license='MIT',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Environment :: Console',
        'Natural Language :: English',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='neural networks,numpy,neural networks implementation'.split(','),
    packages=find_packages(exclude=['tests*']),
    data_files=[('', ['LICENSE'])],
    python_requires='>=3.8, <4',
    install_requires=['numpy'],
)