from setuptools import setup, find_packages

setup(
    name='icnet_tf2',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/icnet_tf2",
    licence="LGPL",
    python_requires='>=3.6',
    description="My tensorflow 2.x ICNet implementation",
    version='1.1.0',
    packages=find_packages(),
    install_requires=[
        "tensorflow"
    ],
)
