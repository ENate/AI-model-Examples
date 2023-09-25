## Concepts

-----------------------------------------------
We discuss the advent of transformers and their applications to training various machine learning problems. To begin, we highlight the steps and the evolution of prior deep learning architectures and their limitations in training.  
### Tools and Tech Stack
- Python 3.10+
- Observation via Grafana (UI), Loki(logs), tempo(traces) and Prometheus (metrics).
- Pytorch (for some examples)
- Examples implemented using mlflow for monitoring and used in training pipelines.
- TensorFlow - a library from training machine learning models
- Flax is a  flexible user experience library via JAX 
### Models 
The following models are implemented in this folder:

### In the begining..
As we know, deep learning neural networks are known to exhibit universal approximation abilites in predicting or classifying problems. However, their limitations in translation tasks, image processing and similar problems have been widely encountered and discussed in literature. Hence, improvements on DNNs have resulted in other types of architectures. For instance, recurrent neural networks (RNNs) -- with a special case of Long Short Term Memory networks (LSTMs), convolutional neural networks (CNNs) and more. Even these architectures have shown remarkable results in translation tasks, image processing, segmentation, speech recognition tasks, they are limited in a number of applications. 
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
