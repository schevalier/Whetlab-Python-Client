import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
	name='whetlab',
	version='',
	description='whetlab-api API library client for python',
	author='Jasper Snoek',
	author_email='jaspersnoek@gmail.com',
	url='http://www.whetlab.com/',
	license='Harvard',
	install_requires=[
		'requests >= 2.1.0'
	],
	packages=[
		'whetlab_api'
	],
	classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: Harvard License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 2.6',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3.2',
		'Programming Language :: Python :: 3.3',
		'Topic :: Software Development :: Libraries :: Python Modules',
	]
)
