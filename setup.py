from setuptools import setup, find_packages

if __name__ == "__main__":
  setup(
    name="samplersLib",
    author="Ahmed H. Bayoumy",
    author_email="ahmed.bayoumy@mail.mcgill.ca",
    version='2601.0',
    packages=find_packages(include=['samplersLib', 'samplersLib.*']),
    description="A samplers library for (in)active sampling that supports both supports both {{online and offline interactions}} with processes executed by external libraries",
    install_requires=[
      'numpy',
      'pyDOE2',
      'scipy',
      'setuptools>=58.1.0',
      'requests',
      'pandas'
      ],
      extras_require={
          'interactive': ['matplotlib>=3.5.2', 'plotly>=5.14.1'],
      },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Intended Audience :: Developers',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
  )