### Introduction

This repository contains examples, tutorials, tools and frameworks on how to prepare training environments, train and apply machine learning (ML) models to problems in various settings. 
The training environments and selected tools will be based on popularity and personal choice with focus on their overall performance. Emphasis will also be laid on examples where performace will affect model training and implementation. Though my aim is not to recommend any particular tools and frameworks but I am hopeful that you may gain from my personal experience in using these tools. Besides, I will also select a number of ML model examples which are mostly suited to my use cases. Specifically, I will mention why using a particular tool may be suitable in a given scenario. Next, I will begin by listing the main tools and discuss the training methods and application of the ML models o interest.

## Preparing the Training Environment 
In order to begin training or fine-tuning any model, we must prepare the training environment. This is necessary in order to facilitate training  and manage different Python versions. This also provides a virtual representation of the libraries and enables us to effectively manage tools and frameworks. It also helps us to prevent potential issues that may arise with using incompatible tools or frameworks which may affect the operating system.

### Technology Stack

  The following tech stack listed below will be used in the folders. 
  I will continue to add to this list on a rolling basis as the need arise. 
  In order to run the examples presented in this repository, I will be using the following tools and frameworks
- Python 3.9+ is available ([by clicking here.](https://www.python.org/))
- Pytorch can be [found here](https://pytorch.org/) 
- TensorFlow can be [found on this link](https://www.tensorflow.org/)
- FLAX is a flexible API and built on JAX [is available.](https://github.com/google/flax)
- More popular AI models, tools, frameworks and libraries are available on [Huggingface](https://huggingface.co/)
- Java, C++ will be used where necessary.

Besides, we list the infrastructural tools for Machine Learning security, monitory and deployment next. 

### More Optional Tools for development
- TFX is an end-to-end platform for deploying ML in production [and available here.](https://www.tensorflow.org/tfx)
- Postgres is an open source database engine [available here.](https://www.postgresql.org/)
- Kafka is a distributed event streaming platform [available here.](https://kafka.apache.org/)
- RabbitMQ is a widely available message broker [which can be installed from here.](https://rabbitmq.com/)
- MySQL 
- docker (and compose) is used to containerize applications to share, verify and run everywhere [can be installed from here.](https://www.docker.com/)
- kubernetes (later)
- Observability ([as discussed here](https://grafana.com/grafana/dashboards/16110-fastapi-observability/)
) using Grafana, Tempo, Loki and Prometheus

### Orchestration
- Airflow is available [on this link](https://airflow.apache.org/)
- Prefect can be installed [from this link.](https://www.prefect.io/)
- Mage offers a more focus on AI [and is available here](https://www.mage.ai/)

### Monitoring/DevOps
- Mlflow can be installed from [this link](https://mlflow.org/)
- Minio is an open source s3 bucket for AI/ML and is available[on this link.](https://min.io/)

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

### Contents of the _misc_folder_

- contains deep neural network and epidemiological models (GUI written in python)
- A Deep neural network architecture drawing file.
- Epidemiological models to study the `n=2` strain in a given population implemented in Python. The aim is to analyze the effect of `n=2` disease strains consisting of different variations. The problem solves the case for `n=2` disease strains affecting a given population, which was submitted in partial fulfillment of the award of the Postgraduate Diploma at the African Institute for Mathematical Sciences, Capetown, South Africa.
- A python program implemented to study the simulation of a molecule in the nucleus of an atom.

### Structure of the repo
- Model training folders and source files: supervised, unsupervised, quantum
- docker: docker files to build infrastructure and services
- infra: for observability tools and services


### Other Resources I like:
- Gitpod is a cloud based ready-to-code environment [can be downloaded from here.](https://www.gitpod.io/)
- Lighting AI platform (Still experimenting and reading their offers) [is available here](https://lightning.ai/).
- Render offers you the opportunity to deploy applications [and is available here.](https://render.com/)

More content to be included soon on a rolling basis - please stay tuned :) 
