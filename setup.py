from setuptools import setup, find_packages
import sys

platform_specific_packages = {
    "darwin": ["tensorflow>=2.4"],
    "linux": ["tensorflow>=2.4"],
    "cywin": ["tensorflow>=2.4"],
    "win3D": ["tensorflow>=2.4"],
}

setup(
    name='icnet_tf2',
    author='Gabriel Van Zandycke',
    author_email="gabriel.vanzandycke@hotmail.com",
    url="https://github.com/gabriel-vanzandycke/icnet_tf2",
    licence="LGPL",
    python_requires='>=3.6',
    description="My tensorflow 2.x ICNet implementation",
    version='1.2.0',
    packages=find_packages(),
    install_requires=[
        *platform_specific_packages[sys.platform]
    ],
)
