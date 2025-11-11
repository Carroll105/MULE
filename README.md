# MULE

MULE (**Mutual Exclusion in scRNA-seq**) is a tool for detecting mutually exclusive gene expression patterns in single-cell RNA-seq and spatial transcriptomics data.

## Features
- Calculate gene–gene mutual exclusivity scores
- Support for single-cell and spatial transcriptomics datasets
- Compatible with [Scanpy]`AnnData` format
- Statistical significance testing with FDR correction

## Installation
nd
conda create -n mule python=3.10 -y
conda activate mule
pip install numba>= 0.61.2
pip install networkx>=3.4.2
pip install scanpy>=0.61.2
pip install -U treelib>=1.7.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


## Usage
- 
Please refer to test_code.ipynb

## Output
- Pairwise exclusivity score matrix
- P-values and FDR-adjusted significance

## License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
