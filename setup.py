from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open('requirements.txt', 'r') as requirements:
    setup(
        name='gloro',
        version='0.0.2',
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
        long_description=long_description,
        long_description_content_type='text/markdown'
    )
