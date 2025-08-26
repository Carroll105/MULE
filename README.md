# MULE

MULE (**Mutual Exclusion in scRNA-seq**) is a tool for detecting mutually exclusive gene expression patterns in single-cell RNA-seq and spatial transcriptomics data.

## Features
- Calculate gene–gene mutual exclusivity scores
- Support for single-cell and spatial transcriptomics datasets
- Compatible with [Scanpy]`AnnData` format
- Statistical significance testing with FDR correction

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Carroll105/MULE.git
cd MULE
pip install -r requirements.txt
```

## Usage
- 
Please refer to test_code.ipynb

## Output
- Pairwise exclusivity score matrix
- P-values and FDR-adjusted significance

## License
This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
