from distutils.core import setup

setup(
    name='autoencodeSVJ',
    version='0.1.0',
    author='Luc Le Pottier',
    author_email='luc.lepottier@cern.ch',
    packages=[
        # 'source',
        'autoencodeSVJ',
    #     # 'autoencode.models',
    #     # 'autoencode.utils',
    #     # 'autoencode.skeletons',
        ],
    # scripts=['driver.py'],
    url='https://github.com/luclepot/autoencodeSVJ',
    license='LICENSE.txt',
    description='Autoencoding component of SVJ analysis.',
    long_description="Autoencoding component of SVJ analysis.",
    requires=[
        "keras",
        "sklearn",
        "pandas",
        "matplotlib"
    ],
)