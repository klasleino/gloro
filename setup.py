from setuptools import setup, find_packages

with open('requirements.txt', 'r') as requirements:
    setup(
        name='gloro',
        version='0.0.1',
        install_requires=list(requirements.read().splitlines()),
        packages=find_packages(),
        description=
            'library for training globally-robust neural networks',
        python_requires='>=3.6',
        author='Klas Leino',
        author_email='kleino@cs.cmu.edu',
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: MIT License',
            'Operating System :: OS Independent'],
        long_description='file: README.md',
        long_description_content_type='text/markdown'
    )
