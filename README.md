# Main Contents

The main parts of this repository consists of the following folders:

- `misc_folders` - containing deep neural network and epidemiological models
  In details, we begin by identifying the contents of the ```misc_folders```
folders and the misc_folder consists of supporting files. The ``` misc_folder ``` contains the following files:
  * A Deep neural network architecture drawing file.
  * Epidemiological models to study the `n=2` strain in a given population implemented in Python. The aim is to analyze the effect of `n=2` disease strains consisting of different variations. The problem solves the case for `n=2` disease strains affecting a given population, which was submitted in partial fulfillment of the award of the Postgraduate Diploma at the African Institute for Mathematical Sciences, Capetown, South Africa.
  * A python program implemented to study the simulation of a molecule in the nucleus of an atom.

- `supervised` - containing deep neural networks models and applications
- `unsupervised` - containing models without label data
- `reinforcement` - describing implementation of models with agents
- `Quantum` - discusses concepts in quantum computing, algorithms and deep learning


## Observability 
We use the example from [this repo](https://github.com/blueswen/fastapi-observability) to monitor the entire services contained in this repository. The implementation is based on:

- Tempo (traces)
- Loki(Logs)
- Prometheus(Metrics)

References will be made to progress in the different models implemented in the folders contained in this repository. For instance,
we will present the links to the papers, tutorials and other forms of publications associated with the topics covered in this repository.

## Tech Stack
- Training is based on using python and related packages such as 
  * TensorFlow
  * FLAX - flexible form and built on JAX
  * Transformers
- Observability ([from here](https://grafana.com/grafana/dashboards/16110-fastapi-observability/)
) using Grafana, Tempo, Loki and Prometheus
