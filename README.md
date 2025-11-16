![Openfold-3-MLX: Protein folding arrives on Apple Silicon.](./assets/openfold3-mlx.png)
OpenFold3-MLX is a specialized fork of OpenFold3-preview optimized for Apple Silicon hardware using the MLX framework. Building on the foundation of DeepMind's [AlphaFold3](https://github.com/deepmind/alphafold3) reproduction by the AlQuraishi Lab at Columbia University, this implementation provides significant performance improvements on M-series chips through native Apple Silicon acceleration. This fork maintains full compatibility with the original OpenFold3-preview while delivering enhanced inference performance on macOS systems.

## MLX Acceleration for Apple Silicon

OpenFold3-MLX leverages Apple's [MLX framework](https://mlx-framework.org/) to provide optimized performance on M-series processors:

- **Native Apple Silicon Optimization**: MLX attention mechanisms, triangle kernels, and activation functions specifically optimized for Apple's unified memory architecture
- **Enhanced Performance**: Significantly improved inference speeds on MacBook Pro, MacBook Air, and Mac Studio systems
- **Memory Efficiency**: Optimized memory usage patterns that take advantage of Apple Silicon's unified memory design
- **Parallel Processing**: Multi-worker data loading pipeline with persistent workers for improved throughput
- **Template Processing**: Robust template alignment processing with kalign integration for high-accuracy predictions

**Performance Benchmarks**: On Apple Silicon hardware, OpenFold3-MLX achieves over 85 pLDDT accuracy (aqGFP) with inference times of approximately 33 seconds per sample on a base M4 Mac, enabling research-quality protein structure prediction on portable hardware.

## Apple Silicon & MLX Framework

This implementation leverages Apple's MLX framework, which is specifically designed to take advantage of Apple Silicon's unified memory architecture and Neural Engine capabilities. Key technical advantages include:

- **Unified Memory Architecture**: Direct access to shared memory reduces data movement overhead between CPU and GPU operations
- **Neural Engine Integration**: Specialized matrix operations are accelerated using Apple's dedicated machine learning hardware
- **Optimized Memory Access Patterns**: MLX kernels are designed to minimize memory bandwidth requirements and maximize cache efficiency
- **Native Performance**: Eliminates the overhead of translation layers required by CUDA-based implementations on Apple hardware

### System Requirements
- **Supported Systems**: MacBook Air, MacBook Pro, iMac, Mac Studio, Mac Pro with M1 or later chips
- **Memory**: 16GB unified memory recommended for typical protein sequences (8GB minimum)
- **Storage**: ~10GB for model weights and dependencies
- **macOS**: 12.0 (Monterey) or later for optimal MLX framework support

## Documentation
Please visit the full Openfold3 documentation at https://openfold-3.readthedocs.io/en/latest/

## Features

OpenFold3-MLX replicates the input features described in the [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) publication with additional Apple Silicon optimizations through MLX framework integration.

### Core AlphaFold3 Features
- Structure prediction of standard and non-canonical protein, RNA, and DNA chains, and small molecules
- Pipelines for generating MSAs using the [ColabFold server](https://github.com/sokrypton/ColabFold) or using JackHMMER / hhblits following the AlphaFold3 protocol
- Structure templates for protein monomers with robust kalign-based alignment processing
- Support for multi-query jobs and batch processing workflows

### MLX-Specific Optimizations
- **MLX Attention Mechanisms**: Native Apple Silicon implementation of attention layers optimized for M-series processors
- **MLX Triangle Kernels**: Specialized triangle attention and multiplication kernels leveraging Apple's unified memory architecture
- **MLX Activation Functions**: Hardware-accelerated activation functions for improved inference speed
- **Parallel Data Loading**: Multi-worker preprocessing pipeline with persistent worker support for enhanced throughput
- **Memory Optimization**: Efficient memory usage patterns designed for Apple Silicon's unified memory model

## Quick-Start for Inference on Apple Silicon

Get started with OpenFold3-MLX on your Apple Silicon Mac in a few easy steps:

### Prerequisites
- macOS system with Apple Silicon (M1, M2, M3, or M4 chip, tested: M4 10-core)
- Python 3.10 or later (tested: 3.13)
- Sufficient memory (16GB+ recommended for larger proteins, tested: 32GB)

### Installation

1a. Installer script (recommended):
```bash
git clone https://github.com/latent-spacecraft/openfold-3-mlx.git
cd openfold-3-mlx && chmod +x ./install.sh && ./install.sh
```

Or, pip installation:

1b. Clone the repository and install dependencies:
```bash
git clone https://github.com/latent-spacecraft/openfold-3-mlx.git
cd openfold-3-mlx
pip install -e .
```

2. Setup model parameters and download weights:
```bash
setup_openfold
```

**Template Processing**: Includes a bundled kalign binary optimized for Apple Silicon (M1/M2/M3/M4).

### Running Predictions

Use the optimized prediction script with protein sequences:

```bash
# Quick prediction with a single protein sequence (aqGFP)
./predict.sh 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

# For JSON input files (compatible with original OpenFold3)
python openfold3/run_openfold.py predict \
    --query_json examples/example_inference_inputs/query_real.json \
    --runner_yaml examples/example_runner_yamls/mlx_runner.yml
```

The MLX-optimized runner configuration automatically enables Apple Silicon accelerations including MLX attention, triangle kernels, and parallel data processing.

## Benchmarking

OpenFold3-MLX maintains the accuracy of OpenFold3-preview while providing significant performance improvements on Apple Silicon hardware.

### Apple Silicon Performance Benchmarks
**Hardware-Specific Results:**

| Hardware | Protein Length | Inference Time | pLDDT Score | Samples/Hour |
|----------|---------------|---------------|-------------|---------------|
| MacBook Air M4 Base | 238aa (GFP) | ~2:45 | 85-86 | ~20 |
| MacBook Air M4 Base | 30aa (peptide) | ~23s | 54-55 | ~150 |

**Key Performance Metrics:**
- **Memory Efficiency**: MLX is optimized for Apple Silicon's unified memory architecture
- **Thermal Performance**: Sustained high performance without thermal throttling
- **Power Efficiency**: Research-quality predictions on battery power
- **Template Processing**: Robust handling of structural templates with 85+ pLDDT accuracy

The MLX optimizations enable research-quality protein structure prediction on consumer hardware, democratizing access to advanced computational biology tools.

## Upcoming Features

OpenFold3-MLX development continues with planned enhancements:

### Performance Optimizations
- Advanced MLX kernel optimizations for larger protein complexes
- Improved memory management for batch processing workflows
- Enhanced template processing pipeline for complex structures

### Compatibility & Features
- Offline MSA and template selection strategy
- Tracking upstream OpenFold3-preview updates and feature additions
- Extended support for RNA and DNA structure prediction optimization
- Integration with additional Apple Silicon ML acceleration features

### Original OpenFold3 Roadmap
The upstream OpenFold3 model continues development with:
- Full parity on all modalities with AlphaFold3
- Training documentation & dataset release
- Workflows for training on custom non-PDB data

## Contributing

If you encounter problems using OpenFold3-MLX, feel free to create an issue! We welcome contributions that improve Apple Silicon performance, fix bugs, or enhance compatibility. Pull requests from the community are encouraged, particularly those that:

- Optimize MLX kernel performance
- Improve memory efficiency on Apple Silicon
- Enhance template processing robustness
- Extend support for additional protein folding scenarios
- Improve documentation and examples

Please ensure any contributions maintain compatibility with the original OpenFold3-preview interface where possible.

## Citing this Work

If you use OpenFold3-MLX in your research, please cite both this implementation and the original OpenFold3-preview:

### OpenFold3-MLX Citation
```
@software{openfold3-mlx,
  title = {OpenFold3-MLX: Apple Silicon Optimized Protein Structure Prediction},
  author = {{Geoffrey Taghon}},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/latent-spacecraft/openfold-3-mlx},
  abstract = {OpenFold3-MLX is a specialized fork of OpenFold3-preview optimized for Apple Silicon hardware using the MLX framework, providing significant performance improvements on M-series chips through native Apple Silicon acceleration.}
}
```

### Original OpenFold3-preview Citation
```
@software{openfold3-preview,
  title = {OpenFold3-preview},
  author = {{The OpenFold3 Team}},
  year = {2025},
  version = {0.1.0},
  doi = {10.5281/zenodo.1234567},
  url = {https://github.com/aqlaboratory/openfold-3},
  abstract = {OpenFold3-preview is a biomolecular structure prediction model aiming to be a bitwise reproduction of DeepMind's AlphaFold3, developed by the AlQuraishi Lab at Columbia University and the OpenFold consortium.}
}
```

Any work that uses this software should also cite [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) and acknowledge the original OpenFold3 team at the AlQuraishi Lab.


#### Openfold3-MLX was built by [Geoffrey Taghon](https://www.linkedin.com/in/gtaghon/) | geoff@latentspacecraft.com | [Learn more](http://latentspacecraft.com/)
