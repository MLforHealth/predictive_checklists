from setuptools import setup

setup(name='predictive_checklists',
      version='0.3',
      description='Creating Predictive Checklists Using Integer Programming',
      url='https://github.com/MLforHealth/predictive_checklists',
      author='Haoran Zhang',
      author_email='haoranz@mit.edu',
      license='BSD',
      packages=['IPChecklists'],
      python_requires='>=3.7',
      install_requires= [
        'numpy>=1.19.0',
        'pandas>=1.1.5',
        'scikit-learn>=0.24.1',
        'scipy>=1.4.1',
        'matplotlib',
        'tqdm',
        'imbalanced-learn>=0.8.0',
        'xgboost>=0.90',
        'optbinning>=0.8.0',
        'mip>=1.13.0',
        'openpyxl>=3.0.9'
      ])
