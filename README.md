# MetDecode

Reference-based deconvolution of methylation patterns

---

## Installation

MetDecode is written in Python 3.11. Dependencies can be installed by running:
```bash
pip3 install -r requirements.txt
```

To install MetDecode:
```bash
python3 setup.py install --user
```

## Running the tool

To run the tool in command line, you will have to execute the `run.py` with the following 3 positioning arguments:
- `atlas-filepath`: TSV file containing the reference atlas. For the input file format, please refer to `data/atlas.tsv` for example. Each tissue / cell type has two dedicated columns, namely the number of methylated CpG sites spanned in the marker region, and the total number of CpG sites (both methylation and unmethylated). Each row corresponds to a marker region. The first 3 columns contain respectively the chromosome, start position and end position of each marker region. The file must contain a header of the form: CHROM    START   END TISSUE1_METH    TISSUE1_DEPTH   TISSUE2_METH    ...
- `cfdna-filepath`: TSV file containing the cfDNA samples. The input file format is similar to `atlas-filepath`, please refer to `data/insilico-cfdna.tsv` for example.
- `out-filepath`: Output CSV file. It will contain the estimations for the cell type contributions. Number of rows (excluding the header) will be equal to the number of cfDNA samples, and the number of columns will be equal to the number of tissues / cell types in the reference atlas.  

The following command runs MetDecode on _in-silico_-generated data with default hyper-parameters.

```bash
python3 run.py example-data/atlas.tsv example-data/cfdna.tsv output.csv
```

If an unknown contributor (a tissue / cell type suspected to be present in the cfDNA mixtures but not present 
in the reference atlas) needs to be modelled, this can be specified with the `-n-unknown-tissues` optional argument:

```bash
python3 run.py example-data/atlas.tsv example-data/cfdna.tsv output.csv -n-unknown-tissues 1
```

By default, the sum of cell type proportions is not equal to 1. However, this constraint can be added using an optional argument:
```bash
python3 run.py example-data/atlas.tsv example-data/cfdna.tsv output.csv --sum1
```

Because MetDecode has been designed for sequencing data and is fed with counts as input, one might consider using the coverage as extra information for more accurate deconvolution. Indeed, in the absence of biases, a higher coverage makes the estimation of the corresponding methylation ratio more reliable. However, in the presence of (biological, technical) biases, such assumption does not hold anymore. To disable the modelling of coverage:

```bash
python3 run.py data/insilico-atlas.tsv insilico-cfdna.tsv output.csv --no-coverage
```

Another key feature of MetDecode is its ability to refine the input atlas by unsupervised deconvolution. The inference atlas should be as accurate as the number of input cfDNA samples is large. If this is not the case, you may consider to disable unsupervised deconvolution:

```bash
python3 run.py data/insilico-atlas.tsv insilico-cfdna.tsv output.csv --supervised
```

---

## Reproducing the results from the paper

First, uncompress the largest data files:

```bash
python3 scripts/uncompress-data.py
```

All deconvolution results, except from simulations, are stored in `results/`.
Simulation results are absent from the repository to to their size.

To reproduce the figures from our paper, you need to run the scripts in `scripts/` as described below.
Figures will be saved in `figures/`

### Deconvolution of cfDNA samples

```bash
python3 scripts/analyze-cfdna-results.py
```

#### Deconvolution benchmarking from scratch

To perform deconvolution from scratch, you can use this command for example:

```bash
python3 scripts/benchmark.py metdecode cfdna 30_250bp significant
```

'metdecode' can be replaced by any other method ('nnls', 'qp', 'celfie' or 'cancerlocator').
DMR size '30_250bp' can be replaced by '30_50bp' or '30_100bp'.
Atlas DMR filter 'all-markers' can be replaced by 'balanced' or 'significant'.

To be able to run 'celfie', you need to include the CelFiE script available at:
[https://github.com/christacaggiano/celfie/blob/master/scripts/celfie.py](https://github.com/christacaggiano/celfie/blob/master/scripts/celfie.py)
inside the `metdecode/` folder.

To be able to run 'cancerlocator', you need to include the CancerLocator.jat available at:
[https://github.com/jasminezhoulab/CancerLocator](https://github.com/jasminezhoulab/CancerLocator)
inside the root `MetDecode/` folder.

### Deconvolution of in silico mixes

```bash
python3 analyze-insilico-results.py
```

To perform deconvolution from scratch, you can use this command for example:
```bash
python3 scripts/benchmark.py metdecode all-insilico 30_250bp significant
```

### Comparison between CBC and deconvolution

```bash
python3 scripts/analyze-cbc-results.py
```

To perform deconvolution from scratch, you can use this command for example:
```bash
python3 scripts/benchmark.py metdecode too-cbc 30_250bp significant
```

### Simulations

```bash
python3 scripts/loo-simulation.py
python3 scripts/simulations.py
python3 scripts/make-sim-figures.py
```
