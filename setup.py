"""
Setup script for Multi-Modal Concentration Analysis System
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def read_requirements():
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='concentration-analyzer',
    version='1.0.0',
    description='Multi-Modal Concentration Analysis System using Computer Vision and Deep Learning',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/concentration-analyzer',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800'
        ],
        'gpu': [
            'torch>=1.9.0+cu111',
            'torchvision>=0.10.0+cu111'
        ]
    },
    entry_points={
        'console_scripts': [
            'concentration-analyzer=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Multimedia :: Video :: Capture',
    ],
    keywords='computer-vision deep-learning concentration-analysis eye-tracking head-pose blink-detection',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/concentration-analyzer/issues',
        'Source': 'https://github.com/yourusername/concentration-analyzer',
        'Documentation': 'https://concentration-analyzer.readthedocs.io/',
    },
)


