# Planning

I think Fast.ai's course is a good overview, and a few people have already told me that going through the lectures are generally easy. I might take this as a first pass to the major topics, so I know what are some of the applications. This would be a good way for me to transition into project planning on my new teams.

I think the degree of overlaps between Coursera's Deep Learning & CS 231N might be higher, whereas the later being more rigorous. I am still debating which one to focus on now, so it might be useful to do a few trial runs.

## Organization

I have surveyed different courses in the past two weeks, and here is how I would characterize the courses and materials:

* **Concept Track**: If the goal is to quickly learn the concpets and jargons, I would start with fast.ai -> Coursera DL specialization. fast.ai talks about all the basic concepts and how they are used in a workflow (code flavor). Coursera DL specialization gives the intuition (more math, notation flavor explanation).

* **Math Track**: I would exclude fast.ai (there's basically no math involved). I would recommend Coursera DL specialization -> Stanford CS courses -> Deep Learning book. This order is in increasingly level of difficulty. For me, I love the intuition in Andrew's coursera courses (basic intro with basic math), but I find Stanford's course notes a lot more detailed and more rigorous. I prefer to use Coursera to gain intuition and Stanford's actual classes to learn the details.

* **Coding Track**: I would recommend fast.ai -> Francis Chollet's Keras Book. fast.ai is very practical, but it wraps a lot of Keras code in fast.ai's own library. I think Francis Chollet's treatment is more comprehensive. I would exclude Andrew's coursera exercises, and perhaps jump straight to Stanford's homework or on-the-job projects.

## Progress

* [Fast.ai](http://wiki.fast.ai/index.php/Main_Page): Top-down teaching approach by Jeremy Howard and Rachel Thomas. Specifically, I want to focus on the ones that use Keras (not Pytorch).

  * [Lecture 0](http://wiki.fast.ai/index.php/Lesson_0): The "surprise lecture" during the data institute launch. It was an introduction to convolutions and machine learning for computer vision
  * [Lecture 1](http://wiki.fast.ai/index.php/Lesson_1): Getting started with deep learning tools for computer vision
  * [Lecture 2](http://wiki.fast.ai/index.php/Lesson_2): The basic algorithms underlying deep learning
  * [Lecture 3](http://wiki.fast.ai/index.php/Lesson_3): New concepts I learned - data agumentation (to make the data richer), dropout (to avoid overfitting, where weights are randomly dropped), Batch normalization (where gradients are normalized every few batches). 

* [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning): Bottom-up teaching approach
	* [Course 1: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
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

	* [Course 2: Improving Deep NN: Hyperparameter Tuning, Regularization, and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome)
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

* [Stanford CS 231N: Convolutional Neural Network](http://cs231n.stanford.edu/syllabus.html): This is highly recommended by Jiaying and Heerad. This is probably also the most rigorous course of the three mentioned here because it's an actual Stanford Course. I was told the Homeworks are superb and I should definitely do them.

  * [Lecture 1](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=1): Course Introduction, nothing substantial
  * [Lecture 2](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=2): Introduction to K-nn, Linear Classifier, nothing that I don't already know.
  * [Lecture 3](https://www.youtube.com/watch?v=h7iBpEHGVNc&index=3&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv): Loss Functions and Optimizations, nothing that I don't already know.
  * [Lecture 4](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=4): Basic intorduction to back propogation


* [Deep Learning with Python](file:///Users/rchang/Downloads/deep_learning_with_python_chollet.pdf)
  * Chapter 1: Introduction to history of AI, ML, and DL
  * Chapter 2: Basics of tensors (multi-dimensional arrays), tensor operations
  * Chapter 3: Anatomy of a neural network, Intro to Keras, 3 examples: binary classification, multi-class classification, and regression, all built using Neural Nets
  * Chapter 4: Workflow of Machine Learning (meta chapter)
  * Chapter 5: This chapter covers CNN, and talked about
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
 
