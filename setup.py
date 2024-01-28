from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="samplersLib",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='24.1.2',
    packages=find_packages(include=['samplersLib', 'samplersLib.*']),
    description="A samplers library for online active sampling",
    install_requires=[
      'numpy>=1.22.4',
      'pyDOE2>=1.3.0',
      'scipy>=1.8.1',
      'setuptools>=58.1.0',
      'requests>=2.20.0',
      'seaborn'
    ],
    extras_require={
        'interactive': ['matplotlib>=3.5.2', 'plotly>=5.14.1'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Intended Audience :: Developers',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
  )