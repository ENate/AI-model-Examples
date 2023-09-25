### Introduction

The repository provides  a theoretical and practical guide on how to prepare environments, train and apply several machine learning models to problems in different settings. We begin with how to prepare a training environment using the most popular technology stacks. Most of the implementations and examples discussed in the repository are done with the proposed tools. However, this is not a recommendation for a particular tool but a matter of choice, convenience and performance. In specific cases, I will mention why using a particular tool may be suitable in a given scenario. For now, I will begin by listing the main tools and discuss the contents of the repository.

### Technology Stack
- Training of the models discussed in this repository is done using
  * TensorFlow
  * Pytorch
  * FLAX - flexible API and built on JAX
  * Many python based deep learning frameworks and libraries
  * Use of Other languages will be highlighted where necessary.
- Observability ([as discussed here](https://grafana.com/grafana/dashboards/16110-fastapi-observability/)
) using Grafana, Tempo, Loki and Prometheus


### Contents

Outline of the main folders contain the following

### supervised
- containing deep neural networks models and applications
### unsupervised 
 In this repository, we discuss examples of models without output labels. In relation to this, we also present examples of problems where non classical training approaches. Note that the main different between unsupervised and supervised learning models is based on the absence of output labels associated with the data corresponding to the proble. Hence, we must distinguish the availability of output labels in the training data before proceeding with making the choice of the training algorithm.
### reinforcement
 - describing implementation of models with agents
### Quantum 
- discusses concepts in quantum computing, algorithms and deep learning


## Observability 
We use the example from [this repo](https://github.com/blueswen/fastapi-observability) to monitor the entire services contained in this repository. The implementation is based on:

- Tempo (traces)
- Loki(Logs)
- Prometheus(Metrics)

References will be made to progress in the different models implemented in the folders contained in this repository. For instance,
we will present the links to the papers, tutorials and other forms of publications associated with the topics covered in this repository.

### misc_folders 
- containing deep neural network and epidemiological models
  In details, we begin by identifying the contents of the ```misc_folders```
folders and the misc_folder consists of supporting files. The ``` misc_folder ``` contains the following files:
  * A Deep neural network architecture drawing file.
  * Epidemiological models to study the `n=2` strain in a given population implemented in Python. The aim is to analyze the effect of `n=2` disease strains consisting of different variations. The problem solves the case for `n=2` disease strains affecting a given population, which was submitted in partial fulfillment of the award of the Postgraduate Diploma at the African Institute for Mathematical Sciences, Capetown, South Africa.
  * A python program implemented to study the simulation of a molecule in the nucleus of an atom.


## Structure of the repository
