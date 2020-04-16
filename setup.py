# -*- encoding: utf-8 -*-
import setuptools


def read_file(file_name):
    with open(file_name, encoding='utf-8') as fh:
        text = fh.read()
    return text


setuptools.setup(
    name='TrajectoryParser',
    author_email='muelleph@cs.uni-freiburg.de',
    description='Tool for parsing optimization trajectories of SMAC and BOHB',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    version=read_file('trajectory_parser/__version__.py').split()[-1].strip('\''),
    packages=setuptools.find_packages(exclude=['*.tests', '*.tests.*',
                                               'tests.*', 'tests'],),
    python_requires='>=3.6, <3.8',
    install_requires=read_file('./requirements.txt').split('\n'),
    test_suite='pytest',
    platforms=['Linux'],
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
