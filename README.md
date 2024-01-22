### Introduction

We present guides on how to prepare training environments, train and apply machine learning models to problems in different settings. 
The training environments will be based on using the most popular technology stacks. Also, the technology stack used in these is based on personal choice and do not focus on their overall performance abilities. However, emphasis will be laid on cases where performace will affet training and implementation of the examples. In relation to this, our natural choice of tools and frameworks will impliitly be  is not a recommended where necessary. Besides performane, I will also select a number of examples based on convenience. In specific cases, I will mention why using a particular tool may be suitable in a given scenario. For now, I will begin by listing the main tools and discuss the training methods and appliation of the resulting models.

## Technology Stack

  The tech stack used in model training involves (not restricted to) the following

- Pytorch
- TensorFlow
- FLAX - flexible API and built on JAX
- More popular AI tools, frameworks and libraries
- Java, C++ will be used where necessary.

Besides, we list the infrastructural tools for Machine Learning security, monitory and deployment next. 

### Infrastructural tools

- Apache Airflow
- MLFLOW
- TFX (with Apache Airflow for orchestration)
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

- examples describing implementation of models with agents.

### Quantum ML models

- examples and resources in applying deep learning models in quantum computing.

## Observability

using the example from [this repo](https://github.com/blueswen/fastapi-observability) to monitor the entire services based on the following stack

- Tempo (traces)
- Loki(Logs)
- Zipkin 
- Prometheus(Metrics)

References will be made to progress in the different models implemented in the folders contained in this repository. For instance,
we will present the links to the papers, tutorials and other forms of publications associated with the topics covered in this repository.

### Contents of the misc_folders

- contains deep neural network and epidemiological models (GUI written in python)
- A Deep neural network architecture drawing file.
- Epidemiological models to study the `n=2` strain in a given population implemented in Python. The aim is to analyze the effect of `n=2` disease strains consisting of different variations. The problem solves the case for `n=2` disease strains affecting a given population, which was submitted in partial fulfillment of the award of the Postgraduate Diploma at the African Institute for Mathematical Sciences, Capetown, South Africa.
- A python program implemented to study the simulation of a molecule in the nucleus of an atom.

## Structure of the repository
