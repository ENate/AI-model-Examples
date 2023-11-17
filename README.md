### Introduction

We present essential guides on how to prepare training environments, train and apply the resulting machine learning models to problems in different settings. The environments presented in this guide will be based on the most popular technology stacks. Also, the associated technology stack discussed is based on personal choice and experience with emphasis based on performance issues. However, we hope to identify situations where performace will be perceived to affect training and implementation of the examples discussed. In relation to this, our natural choice of tools and frameworks will implicitly be recommended where necessary. Besides performance, I will also select a number of examples based on convenience (due to the availability of tools and datasets). In specific cases, I will mention why using a particular tool may be suitable in a given scenario. For now, I will begin by listing the main items contained in the chosen technology stack, and then discuss the training methods and appliation of the resulting models.

### Technology Stack

  The tech stack used in model training involves (not restricted to) the following

- Pytorch
- TensorFlow
- FLAX - flexible API and built on JAX
- Python AI tools, frameworks and libraries
- Java, C++ will be used where necessary.

Besides, we list the infrastructural tools for Machine Learning security, monitory and deployment next. 

### Infrastructural tools

- Apache Airflow
- MLFLOW
- TFX (with Apache Airflow for orchectration)
- Postgres
- Kafka
- RabbitMQ

- Observability ([as discussed here](https://grafana.com/grafana/dashboards/16110-fastapi-observability/)
) using Grafana, Tempo, Loki and Prometheus

### Contents

Outline of the main folders contain the following

### supervised

- containing deep neural networks models and
- base-trainer
- generative-ai contains descriptive models on foundational and practical model applications in generative problems.

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
  - A Deep neural network architecture drawing file.
  - Epidemiological models to study the `n=2` strain in a given population implemented in Python. The aim is to analyze the effect of `n=2` disease strains consisting of different variations. The problem solves the case for `n=2` disease strains affecting a given population, which was submitted in partial fulfillment of the award of the Postgraduate Diploma at the African Institute for Mathematical Sciences, Capetown, South Africa.
  - A python program implemented to study the simulation of a molecule in the nucleus of an atom.

## Structure of the repository
