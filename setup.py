from setuptools import setup, find_packages

setup(
    name="i308-calib",
    version="0.0.1",
    packages=find_packages(),
    #include_package_data=True,
    package_data={"i308_calib": ["cfg/*.yaml"]},
    install_requires=[
        "opencv-python",
        "matplotlib",
        "pyyaml"
    ],

    entry_points={
        'console_scripts': [
            'calib = i308_calib.tool_mono:run',
            'calib-stereo = i308_calib.tool_stereo:run',
        ],
    },
    author="Esteban Uriza",
    description="camera calibration tool",
    url="https://github.com/udesa-vision/i308-calib",
)
