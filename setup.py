import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
	name='whetlab',
	version='0.1',
	description='Whetlab client for Python',
        long_description=open('README.md').read(),
	author='Whetlab LLC',
	author_email='info@whetlab.com',
	url='http://www.whetlab.com/',
	license='LICENSE.txt',
	install_requires=[
		'requests >= 2.1.0'
	],
	packages=[
		'whetlab',
                'whetlab.server',
                'whetlab.server.api',
                'whetlab.server.error',
                'whetlab.server.http_client'
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
