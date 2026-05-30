from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"

INSTALL_REQUIRES = [
    "ultralytics==8.3.221",
    "scikit-image==0.25.2",
    "tifffile==2022.8.12",
    "onnxruntime==1.23.2",
    "segmentation-models-pytorch==0.5.0",
    "h5py==3.15.1",
    "future==1.0.0",
    "scikit-learn==1.7.2",
    "ipyparallel==9.0.2",
    "holoviews==1.21.0",
    "numba==0.62.1",
    "olefile==0.47",
    "Flask==3.0.0",
    "SQLAlchemy==2.0.44",
    "zarr==2.18.3",
    "hydra-core==1.3.2",
    "opencv-python==4.8.0.74",
    "numpy==1.26.4",
    "matplotlib==3.10.7",
    "tqdm==4.67.1",
    "torch==2.1.1+cu118",
    "torchvision==0.16.1+cu118",
    "scipy==1.15.3",
    "pillow==11.3.0",
    "PyYAML==6.0.3",
    "omegaconf==2.3.0",
]


setup(
    name="Airscope-ca",
    version="0.1.0",
    description="PICO calcium imaging processing pipeline",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.11",
    license="GPL-3.0-only",
    author="Yuanlong Zhang, Angran Li, Lekang Yuan, Mingrui Wang",
    keywords=[
        "calcium imaging",
        "mesoscope",
        "neuroscience",
        "Airscope",
        "motion correction",
        "segmentation",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    packages=find_packages(include=["Airscope_ca", "Airscope_ca.*"]),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "airscope-process=Airscope_ca.process_script:main",
        ],
    },
)
