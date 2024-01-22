### Concepts

-----------------------------------------------

We discuss transformer architectures and their applications to various machine learning problems. To begin, we highlight the 
the limitation of prior training models which were inefficient to deal with a 
number of issues associated with Machine Translation, Document and Text Mining, Computer Vision 
and other Natural Language Processing tasks. Next, we identify the main contribution of transformer
models in solving such defects in these problems. 
To proceed, we implement an example of a transformer from scratch and identify the 
steps associated with applying them in deep learning.  In order to follow this tutorial and codes in this folder,
the following steps will be discussed

### Main Items
- Attention Mechanism in Transformers
- Building the transformer architecture
- Discussing the main points to consider when training a transformer model
- Pre-training and the associated tools

### Tools and Tech Stack

- Python 3.10+
- Observation via Grafana (UI), Loki(logs), tempo(traces) and Prometheus (metrics).s
- Pytorch (for some examples)
- Examples implemented using mlflow for monitoring and used in training pipelines.
- TensorFlow - a library from training machine learning models
- Flax is a  flexible user experience library via JAX

### Models

The following models are implemented in this folder:

### In the begining

We assume familiarity with the ability of deep learning neural networks to act as universal approximators  in predicting or classifying problems. However, their limitations in translation tasks, image processing and similar problems have been widely encountered and discussed in literature. 
Hence, improvements on DNNs have resulted in other types of architectures. For instance, recurrent neural networks (RNNs) -- with a special case of Long Short Term Memory networks (LSTMs), convolutional neural networks (CNNs) and more. Even these architectures have shown remarkable results in translation tasks, image processing, segmentation, speech recognition tasks, they are limited in a number of applications.
Language models represent supervised learning models used to train and develop text and document based learning.

### Life before BERT

- Bidirectional Encoder Representations from Transformers (BERT)
- Use of machine language translation
- Attention based models via the `Àttention is All you Need` paper

### BERT Model

- Based on introduction of optimal training to model

### Life After BERT

- Usage of `simplified` training architecture
- Removed bottlenecks in training
- Added more simplified attention based parts in training
- Rise in large language models with billions of parameters
