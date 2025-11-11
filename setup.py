from setuptools import setup, find_packages
setup(
    name="mule",  # The name of the package
    version="0.1.0",  # The version of the package
    packages=find_packages(),  # Automatically discover and include all packages in the project
    install_requires=[  # External dependencies required for the project
        "numpy>=2.2.6",  # Required version of numpy
        "pandas>=2.3.2",  # Required version of pandas
        "scanpy>=1.11.4",  # Required version of scanpy
        "torch>=2.5.1",  
        "networkx>=3.4.2",
        "treelib>=1.7.1",
        "numba>=0.61.2",
    ],
    python_requires=">=3.10",  # Python version requirement
    author="Jinpu Cai",  # Your name
    author_email="jinpucai99@gmail.com",  # Your email address
    description="MULE:Mutual Exclusion in scRNA-seq",  # A short description of the package
    long_description="""MULE (Mutual Exclusion in scRNA-seq) is a tool for detecting mutually exclusive gene expression patterns in single-cell RNA-seq and spatial transcriptomics data.""",  # A detailed description of the package
    long_description_content_type="text/plain",  # The format of the long description (plain text)
    url="https://github.com/Carroll105/MULE",  # URL to the project homepage or GitHub repository
    classifiers=[  # Classifiers to describe the project (useful for the package index)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="single-cell RNA-seq,Mutual Exclusion",  # Keywords related to the project for searchability
)
