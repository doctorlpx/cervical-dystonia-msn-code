# Cervical dystonia morphometric similarity networks

Python and MATLAB code for regional morphometric similarity, imaging transcriptomics, gene-set enrichment and neurotransmitter receptor analyses in cervical dystonia.

## Run the analysis

Install the Python dependencies, prepare the inputs locally using the [preparation instructions](preparation/README.md), then run:

```sh
python -m pip install -r requirements.txt
python run_analysis.py --inputs prepared --output results
```

The default run fits the primary Schaefer-400 model and both sensitivity analyses. It writes results to a new or empty directory. Use `--help` to select an individual analysis or to supply existing group maps.

MRI data come from [OpenNeuro ds006395, version 1.0.8](https://doi.org/10.18112/openneuro.ds006395.v1.0.8). Molecular inputs come from the [Allen Human Brain Atlas](https://human.brain-map.org/static/download) and the [Hansen et al. receptor atlas](https://github.com/netneurolab/hansen_receptors).

