from setuptools import setup, find_packages

setup(
    name='boat-tracking-system',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A system for detecting and tracking boats in video footage using YOLO.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'numpy',
        'ultralytics',
        'matplotlib',
        'PyYAML',
        'loguru',
    ],
    entry_points={
        'console_scripts': [
            'boat-tracking=main:main',  # Adjust this based on your main function location
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)