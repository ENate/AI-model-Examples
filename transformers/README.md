# Introduction

-----------------------------------------------

The contents of this folder consist of three types of transformer based architectures: encoder, decoder and encoder-decoder models. The folder is structured between implementing the aforementioned transformer models in two stages. In the first instance, I basically copied most of the classical encoder-decoder example implemented from scratch. And then implemented both the encoder and decoder transformer based models/architectures to present their implementation in a simplified manner.
In doing so, we highlight why  transformer models have become so popular in various applications by identifying their unique performance in numerous tasks. Transformer based models differ from other classical AI models due to the development of Attention Mechanisms (local and global context) and novelty in training (for ease of parallelization). I will also discuss why there points introduced bottlenecks when sequence to sequence models were used in Machine Translation,  and other deep learning architectures in Document, Text Mining, Computer Vision and other tasks. 
More so than that, it has also been adapted for different problem types.
Next, we identify the main contribution of transformer models in solving such defects in these problems. 
To proceed, we implement an example of a transformer from scratch and identify the 
steps associated with applying them in deep learning.  In order to follow this tutorial and codes in this folder,
the following steps will be discussed

## Main Items
This sections is divided into two main folders with each folder consisting of 3 types of transformer architectures. A brief description contains the following:

- fine-tuned consists of 
1) encoder
2) decoder
3) encoder-decoder

- The ```from_scratch``` consists of 
1) encoder
2) decoder
3) encoder-decoder

In each case, we validate the models with various data types of various compositions.

We highlight the main items covered in this folder. First, I will present a summary on how to build 
a transformer from scratch. Using some of the tutorials and tools that are available, I also use some code examples
without a deep dive into the implementations. My goal is to give you more control over your desire to study the transformer architecture.
I also feel that by studying using these explanations (mine included :)), you can pick up more ideas from these numerous sources.
To begin, we will cover the following topics:

- Discuss Attention Mechanism in Transformers
- Building the transformer architecture from Scratch
- Discussing the main points to consider when training a transformer model
- Pre-training Transformer Models including Tools and Frameworks

### Implementation Steps

- Initial text pre-processing
- Embedding preprocessed text
- Pass text to encoder
- Decode text
- Pass to encoder-Decoder Attention
- Training and Output Analysis

### Tools and Tech Stack Required in This Folder

- Python 3.10+
- Airflow, Mage or Prefect
- Pytorch - a library for training ML models
- Mlflow for monitoring and used in training pipelines.
- TensorFlow - a library from training machine learning models
- Flax is a  flexible user experience library via JAX

### Models

The models implemented in these folders can be categorized into ```from_scratch```  and fine-tuned models. The former case involves
implementing these models from scratch. Implementing models from scratch definitely gives you more control and the opportunity
to understand the whole model architecture. Besides giving you a much better control of working with these models, it also
makes it easy for you to clearly understand models during fine-tuning. Model fine-tuning gives you even more 
opportunity to use ready-made models (somewhat) which are trained with large data sets (which may not be available for you) to build your own model.
It is also said that fine-tuning models will reduce your carbon footprint (we should really care about this), and given you state-of-the-art models
to work with. Please check for more benefits of working with pre-trained models.

### In the Beginning

We assume familiarity with deep learning neural networks (which are known to be universal approximators)
in predicting or classifying problems. However, to add the limitations encountered in machine translation tasks, 
image processing and identify problems which had been widely encountered as discussed in literature. 
Hence, improvements on DNNs have led to their adaptation and usage in other types of architectures. 
For instance, recurrent neural networks (RNNs) -- with a special case of Long Short Term Memory networks (LSTMs), convolutional neural networks (CNNs) and more. Even these architectures have shown remarkable results in translation tasks, image processing, segmentation, speech recognition tasks, they are limited in a number of applications.
Language models represent supervised learning models used to train and develop text and document based learning.

### Life before BERT

- Construction of recurrent neural networks (RNNs)
- Introduction of Long Short Term Memory models
- Use of machine language translation with different RNNs
- Attention based models via the use of sequence-to-sequence models


### BERT Model

- Bidirectional Encoder Representations from Transformers (BERT)
- The arrival of the _Attention is All you Need_ Paper
- Based on introduction of optimal training to model

### Life After BERT

- Usage of _simplified_ training architecture
- Removed bottlenecks in training
- Added more simplified attention based parts in training
- Rise in large language models with billions of parameters 
- Fine-tuning to save the day
- Cost, applications and the AI ambush (mostly in documents, text and Video) in applications

Note: The contents will continuously be updated with better writing techniques on a rolling basis.
