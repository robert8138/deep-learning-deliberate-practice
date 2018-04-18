# Progress

For this deliberate practice, I ended up doing something different - instead of learning **MULTIPLE concepts with a SINGLE source of book/material/MOOC**, I am trying out to **learn a SINGLE concept from MULTIPLE sources**. This is something that I have always wanted to do in graduate school (i.e. relearning the same concept multiple times, but in different ways), and I believe it's the right thing to do. It also happened that there are a lot of reputable DL materials at my disposal, and I have been reading and learning them all at once without too much trouble. So far, I enjoyed learning about the same concept from different angles, and I think it has the side benefit of [interleaving](https://www.scientificamerican.com/article/the-interleaving-effect-mixing-it-up-boosts-learning/).

## Topics

* Basics of Neural Network, Mathematical set up and formulation
* Loss Function, Optimization routines, Hyperparameter tuning
* How Forward Pass and Backward Propogation works
* Practical Advice for building ML Projects
* Convolutional Neural Network, image-based models
* Visualizing CNNs
* Recurrent Neural Network, Sequence Model, text-based models (**Deliberately leave this out**)
* Unsupervised Learning Techniques or Generative Models

I am deliberately leaving out RNN related stuff in fast.ai, Stanford course, Keras book, and Coursera DL specialization, because I want to focus on CNN for now. I will definitely revisit when I have enough grasp of CNNs.

## Organization

I have surveyed different courses in the past two weeks, and here is how I would characterize the courses and materials:

* **Concept Track**: If the goal is to quickly learn the concpets and jargons, I would start with fast.ai -> Coursera DL specialization. fast.ai talks about all the basic concepts and how they are used in a workflow (code flavor). Coursera DL specialization gives the intuition (more math, notation flavor explanation).

* **Math Track**: I would exclude fast.ai (there's basically no math involved). I would recommend Coursera DL specialization -> Stanford CS courses -> Deep Learning book. This order is in increasingly level of difficulty. For me, I love the intuition in Andrew's coursera courses (basic intro with basic math), but I find Stanford's course notes a lot more detailed and more rigorous. I prefer to use Coursera to gain intuition and Stanford's actual classes to learn the details.

* **Coding Track**: I would recommend fast.ai -> Francis Chollet's Keras Book. fast.ai is very practical, but it wraps a lot of Keras code in fast.ai's own library. I think Francis Chollet's treatment is more comprehensive. I would exclude Andrew's coursera exercises, and perhaps jump straight to Stanford's homework or on-the-job projects.

## Progress

### University Courses

* [Fast.ai](http://wiki.fast.ai/index.php/Main_Page): Top-down teaching approach by Jeremy Howard and Rachel Thomas. Specifically, I want to focus on the ones that use Keras (not Pytorch).

  * [Lecture 0](http://wiki.fast.ai/index.php/Lesson_0): The "surprise lecture" during the data institute launch. It was an introduction to convolutions and machine learning for computer vision
  * [Lecture 1](http://wiki.fast.ai/index.php/Lesson_1): Getting started with deep learning tools for computer vision
  * [Lecture 2](http://wiki.fast.ai/index.php/Lesson_2): The basic algorithms underlying deep learning
  * [Lecture 3](http://wiki.fast.ai/index.php/Lesson_3): New concepts I learned - data agumentation (to make the data richer), dropout (to avoid overfitting, where weights are randomly dropped), Batch normalization (where gradients are normalized every few batches).
  * [Lecture 4](https://www.youtube.com/watch?v=V2h3IOBDvrA&feature=youtu.be): Mainly about optimization techniques - momentum, RMSprop, ADAM. 
  * [Lecture 5](http://wiki.fast.ai/index.php/Lesson_5): Adding batchnorm to VGG; visualizing latent factors; functional API (skipped RNN stuff for now)
  * **TODO**: [Lecture 6](http://wiki.fast.ai/index.php/Lesson_6): Finished the first hour (psuedo-labeling, word embedding visualized again)
  * [Lecture 7](http://wiki.fast.ai/index.php/Lesson_7): CNN architectures: resnet, inception, fully convolutional net, multi input and multi output nets; localization with bounding box models and heatmaps; using larger inputs to CNNs (skip RNN stuff for now)

* [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning): Bottom-up teaching approach
	* **DONE**: [Course 1: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
		* Week 1: 
			- Understand the major trends driving the rise of deep learning. 
			- Be able to explain how deep learning is applied to supervised learning. 
			- Understand what are the major categories of models (such as CNNs and RNNs), and when they should be applied. 
			- Be able to recognize the basics of when deep learning will (or will not) work well.
		* Week 2: 
			- Build a logistic regression model, structured as a shallow neural network. 
			- Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent. 
			- Implement computationally efficient, highly vectorized, versions of models.
			- Understand how to compute derivatives for logistic regression, using a backpropagation mindset.
		* Week 3: 
			- Understand hidden units and hidden layers. 
			- Be able to apply a variety of activation functions in a neural network.
			- Build your first forward and backward propagation with a hidden layer.
			- Apply random initialization to your neural network.
			- Become fluent with Deep Learning notations and Neural Network Representations.
			- Build and train a neural network with one hidden layer.
		* Week 4: 
			- See deep neural networks as successive blocks put one after each other
			- Build and train a deep L-layer Neural Network.
			- Analyze matrix and vector dimensions to check neural network implementations.
			- Understand how to use a cache to pass information from forward propagation to back propagation.
			- Understand the role of hyperparameters in deep learning

	* **DONE**: [Course 2: Improving Deep NN: Hyperparameter Tuning, Regularization, and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome)
		* Week 1:
			- Recall that different types of initializations lead to different results
			- Recognize the importance of initialization in complex neural networks
			- Recognize the difference between train/dev/test sets
			- Diagnose the bias and variance issues in your model
			- Learn when and how to use regularization methods such as dropout or L2 regularization
			- Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
			- Use gradient checking to verify the correctness of your backpropagation implementation
		* Week 2:
			- Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
			- Use random minibatches to accelerate the convergence and improve the optimization
			- Know the benefits of learning rate decay and apply it to your optimization
		* Week 3:
			- Master the process of hyperparameter tuning

	* **DONE**: [Course 3: Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects/home/week/1)
		* Week 1:
			- Understand the importance of ML strategy for project/product iteration
			- The concept of orthogonalization (doing one thing at a time, early stopping is an anti-example)
			- Single evaluation metric (for each of comparison across models)
			- Satisficing and Optimizing metrics (one metric for optimization, and rest as constraints that need to be satisfied)
			- Train/Dev/Test (99%/1%/1% is appropriate for DL)
			- Make sure dev/test reflects the underlying true distribution
			- For perception based task, very important to use human-level performance as a proxy for bayes error. 
				- Delta between `human-level performance` and `training error` is called "avoidable bias"
				- Delta between `training error` and `dev set error` is called "variance"
		* Week 2:
			- For ML project, always build something quick and dirty, then reiterate
			- Doing manual error analysis to understand why ML model is not doing well
			- In DL era, be aware of the situations where training distribution != dev/test distribution
			- When diagnosing model performance: have the folllowing metrics
				- Human-level performance
				- Training set error (If training error >> human-level performance, it's "avoidable bias")
				- Training-dev set error (If training error >> training-dev error, it's "variance problem")
				- Dev set error (If training error ~ training-dev error >> dev set error, it's data distribution mismatch)
				- Test set error (If test error >> dev error, you might have overfitted dev set)
			- When there is data distribution mismatch, try to make training data as similar as dev/test as possible
			- When to use Transfer Learning (from A -> B):
				- Lower level concepts from A can be re-used for task B
				- There's a lot more data in task A than in task B
				- Task A and B have the same input
			- When to use Multi-task Learning (common in object detection):
				- Training on a set of tasks can all shared useful low level features
				- Usually, the amount of data required is about the same (so can combine forces)
				- Can train a big enough network to do well on the tasks at once
			- End-to-End Deep Learning
				- Pros: let the data speak, less hand-engineering features
				- Cons: may need a large amount of data, exclude potentially useful hand-designed features
				- The trade-off really depends on how much data there is. If a lot of data, less need for hand-designed features. If not a lot of data, hand-designed features are more helpful.

	* **DONE**: [Course 4: Convolutional Neural Network](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
		* Week 1:
			- Understand the convolution operation
			- Understand the pooling operation
			- Understand how fully connected layer works with CNN
			- Remember the vocabulary used in convolutional neural network (padding, stride, filter, ...)
			- Visualize how one layer of convolution/pooling/FC works
			- Visualize how one forward pass of the entire CNN works
		* Week 2:
			- Understand multiple foundational papers of convolutional neural networks (LeNet, AlexNet, VGG 16, ResNet)
			- Analyze the dimensionality reduction of a volume in a very deep network (using Networks in Networks as 1x1 Convolutions)
			- Understand and Implement a Residual network
			- Understand the concept of Inception Network (from the movie inception) and how it uses 1x1 convolutions to reduce multiplication operations
			- Data Augmentation/Transfer learning
			- [Keras Intro](https://github.com/Kulbear/deep-learning-coursera/blob/master/Convolutional%20Neural%20Networks/Keras%20-%20Tutorial%20-%20Happy%20House%20v1.ipynb)
		* Week 3:
			- Understand the challenges of Object Localization, Object Detection and Landmark Finding. Remember the vocabulary of object detection (landmark, anchor, bounding box, grid, ...). Most importantly, Understand how we label a dataset for an object detection application.
				- object localization -> `y = (pc, bounding box center coordinates, bounding box width and heights, one-hot encoding of categories)`
				- landmark -> `y = (pc, [landmark x, landmark y]*m, one-hot encoding)`, m is defined by the number of landmarks we want to track -> this is how AR works
				- object detection -> `y = ([(pc, [landmark x, landmark y]*, one-hot encoding)] * m)`, for m different bounding boxes
			- Understand and implement intersection over union and non-max suppression to get more accurate bounding boxes (these are post processing after predictions are made)
		* Week 4:
			- Intuition Behind the formulation for face verification/recognition problems
				- Siamese Network, Triplet Loss formulation, or Binary classification formulation
			- Intuition Behind the formulation for neural style transfer
				- Cost = Cost for Conent + Cost for Style
	* **DONE**: [Course 5: Sequence Model](https://www.coursera.org/learn/nlp-sequence-models/home/welcome)
		* Week 1:
			- Learn about recurrent neural networks: one-to-one (traditional NN), one-to-many (music generation), many-to-one (sentiment analysis), many-to-many (machine translation)
			- The basic set up of recurrent neural network, forwardprop in time, backprop through time
			- The issue with vanishing gradient (solved by GRU, LSTM), exploding gradient (graident capping)
			- Gated Recurrent Unit (GRU)
			- Long, Short Term Memory (LSTM)
			- Bidirectional RNN (train forward and train backward)
			- Deep RNN (stacking layers of RNN)
		* Week 2:
			- Word representation using one-hot encoding (inefficient, sparse)
			- Word representation using embeddings (efficient, dense)
			- Getting more intuition about embeddings through understanding its property
			- Embedding matrix (loopup table: word index -> word vector representation)
			- Learning word embeddings: word2vec (skip-gram model, negative sampling)
			- Learning word embeddings: GloVe (using co-occurrences matrix of context & target word)
		* Week 3:
			- Basics of seq2seq model (encoder + decoder architecture)
			- Beam search
			- Error analysis of approximate optimization aglorithm on object function
			- Bleu scores
			- Attention model


* [Stanford CS 231N: Convolutional Neural Network](http://cs231n.stanford.edu/syllabus.html): This is highly recommended by Jiaying and Heerad. This is probably also the most rigorous course of the three mentioned here because it's an actual Stanford Course. I was told the Homeworks are superb and I should definitely do them.

  * [Lecture 1](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): Course Introduction, nothing substantial
  * [Lecture 2](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=2): Introduction to K-nn, Linear Classifier, nothing that I don't already know.
  * [Lecture 3](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=3): Loss Functions and Optimizations, nothing that I don't already know.
  * [Lecture 4](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4): Basic intorduction to back propogation
  * [Lecture 5](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5): Basic introduction to convolutional Neural Network. This is a subset of the [CovNet notes](http://cs231n.github.io/convolutional-networks/), which is a lot more comprehensive.
  * [Lecture 6](https://www.youtube.com/watch?v=wEoyxE0GP2M&index=6&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): It's kind of funny how this lecture came after the CNN lecture. Activation functions, initialization, dropout, batch normalization. I would refer to [Neural Net Notes I](http://cs231n.github.io/neural-networks-1/) to learn the details. 
  * [Lecture 7](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=7): Optimization update rules, ensembles, data augmentation, transfer learning. [Neural Net Notes I](http://cs231n.github.io/neural-networks-1/). [Neural Net Notes II](http://cs231n.github.io/neural-networks-2/). [Neural Net Notes III](http://cs231n.github.io/neural-networks-3/).
  * [Lecture 8](https://www.youtube.com/watch?v=6SlgtELqOWc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=8): Introduction to CPU v.s. GPU. Deep Learning software framework, Justin focused on Tensorflow and Pytorch. Last slide on which framework to use is useful.
  * [Lecture 9](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=9): Clearest lecture on all the ImageNet winner models from 2012 - 2016. Introduce AlexNet (2012), VGG (2014), GoogLenet (2015), ResNet (2016).
  * [Lecture 10](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): RNN, LSTM, GRU, word embeddings
  * [Lecture 11](https://www.youtube.com/watch?v=nDPWywWRIRo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=11): Image segmentation, Image localization, object detection, instances detection (all based on CNN as the foundation with different label sets)

* [Stanford CS 224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/archive/WWW_1617/syllabus.html): This is one of the other pillar class for DL at Stanford, focusing on NLP. I originally didn't intend to study this, but I am using it to complement details didn't covered in the Coursera C5 on Sequence Models. Lecture with asterisk are more important
  * [*Lecture 2](https://www.youtube.com/watch?v=ERibwqs9p38&index=2&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6): Introduction to Word2Vec, Predicting outside/context word | center word, Cost function setup. Chris Manning deriving partial derivatives.
  * [*Lecture 3](https://www.youtube.com/watch?v=ASn7ExxLZws&index=3&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6): Yet another word embedding algorithm - GloVec. External & Internal evaluation for word embeddings
  * [Lecture 4](https://www.youtube.com/watch?v=uc2_iwVqrRI&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=4): Backward Propogation
  * [Lecture 5](https://www.youtube.com/watch?v=isPiE-DBagM&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=5): Backward Propogation II
  * [Lecture 7](https://www.youtube.com/watch?v=PicxU81owCs&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6&index=7): Introduction to Tensorflow - computational graph, placeholder, variable, optimizer, session, execution environment
  * [*Lecture 8](https://www.youtube.com/watch?v=Keqep_PKrY8&index=8&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6): Vanilla RNN, Bidirectional RNN, Deep RNN
  * [Lecture 13](https://www.youtube.com/watch?v=Lg6MZw_OOLI&index=14&list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6): CNN on text data

### Toolings: Python & Keras

* [Deep Learning with Python](file:///Users/rchang/Downloads/deep_learning_with_python_chollet.pdf)
  * **Chapter 1**: Introduction to history of AI, ML, and DL
  * **Chapter 2**: Basics of tensors (multi-dimensional arrays), tensor operations
  * **Chapter 3**: Anatomy of a neural network, Intro to Keras, 3 examples: binary classification, multi-class classification, and regression, all built using Neural Nets
  * **Chapter 4**: Workflow of Machine Learning (meta chapter)
  * **Chapter 5**: This chapter covers CNN, and talked about
  	* Convolution, padding, strides, max-pooling
  	* Training a covnet from scratch using small dataset
  	* Data augmentation (enrich data) + Dropout (to avoid overfitting)
  	* Using pre-trained CNN
  		* Feature extraction (with or without data augmentation)
  		* Fine Tuning: specific layers (pop back a few layers and retrain them)
  	* Visualizaing CNN
  		* Visualize activation
  		* Visualize covnet filters
  		* Visualize which of the input pixel contribute to the classification
  * **Chapter 6**:
  	* Word Embeddings (word index -> word vector): going from one-hot-encoding vectors (sparse) to embedding vectors (dense). Essentially, learn a richer representation for relationship among words.
		* [Word2Vec](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf): Skip-gram model & negative sampling to learn embeddings. This class notes has detailed formulation for the objective function, the loss function, and what we need to optimize.
		* [GloVec](http://web.stanford.edu/class/cs224n/lectures/lecture3.pdf): Use (context, target) pair co-occurrences to learn embeddings
		* Leverage pre-trained word embeddings for transfer learning
	* Recurrent Neural Network
		* Simple Recurrent Neural Network
		* Gated Recurrent Unit (GRU)
		* Long, Short Term Memory (LSTM)
	* Additional Techniques to add complexity to RNN
		* Bidrectional Nerual Network (train forward and train backward, prediction made by using both forward and backward activation. See Coursera example)
		* Deep RNN (stacking additional layer for each time t)
	* Sequence Processing using 1D CovNet
  * **Chapter 7**:
  	*  How to use functional APIs to build models as arbitrary graphs of layers (single-/multi- intput -> single-/multi- output), reuse layers (layer weight sharing), and use models as Python functions (model templating)
	* You can use Keras callbacks to monitor your models during training and take action based on model state
	* TensorBoard allows you to visualize metrics, activation histograms, and even embedding spaces
	* how to do batch normalization, depthwise separable convolution, and residual in Keras
	* Why you should use hyperparameter optimization and model ensembling
 
### Textbook

* [Deep Learning Book](http://www.deeplearningbook.org/)
	* [**Chapter** 11](http://www.deeplearningbook.org/contents/guidelines.html): Practical Methodology. Corresponding [video](https://www.youtube.com/watch?v=ccyClyHAIdI).