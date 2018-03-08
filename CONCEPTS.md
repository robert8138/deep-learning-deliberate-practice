# Deep Learning Workflow

* **Defining the problem and assembling a dataset**
	* What will your input data X be? What are you trying to predict (y)?
	* What type of problem are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Something else, like clustering, generation, or reinforcement learning? Identifying the problem type will guide your choice of model architecture, loss function, and so on.
	* Be aware of the hypotheses you make at this stage:
		* You hypothesize that your outputs can be predicted given your inputs
		* You hypothesize that your available data is sufficiently informative to learn the relationship between inputs and outputs

* **Choosing a measure of success**
	* To achieve success, you must define what you mean by success—accuracy? Precision and recall? Customer-retention rate? Your metric for success will guide the choice of a loss function.


* **Deciding on an evaluation protocol**
	* Maintaining a hold-out validation set: The way to go when you have plenty of data
	* Doing K-fold cross-validation: The right choice when you have too few samples for hold-out validation to be reliable
	* Doing iterated K-fold validation: For performing highly accurate model evaluation when little data is available


* **Data pre-processing**
	* Vectorization (reshape them into tensors)
	* Value normalization (between 0 and 1)
	* Handle Missing values

* **Developing a model that does better than a baseline**
	* Decide what is the last layer configuration
	* Loss function
	* Optimizatino routine

* **Develop a model that overfits**
	* Add layers
	* Make the layers bigger
	* Train for more epochs

* **Regularizing your model and tuning your hyperparameters**
	* Data Augmentation
	* Add dropouts
	* Add L1/L2 regularization
	* Add or remove layers

* **Final model**
	* Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it one last time on the test set.


# Backward Propogation

Backward Propogation is one of the most important concept in Deep Learning. Specifically, it describes how we could calculate the gradient of the cost function with respect to the parameters of the neural network. 

The math for backprogation can often time be confusing, so a lot of people have written materials to explain in more details what they are. I particularly find the materials linked from [CS 231N](http://cs231n.stanford.edu/syllabus.html) most helpful, so I will summarize them here.

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/): This post explains how, fundamentally, backpropogation is a technique for calculating derivatives quickly. And it’s an essential trick to have in your bag, not only in deep learning, but in a wide variety of numerical computing situations. It doesn't go into details in explaining the derivation of gradient for DL, but it explains why backprop is a much more efficient way to compute derivatives than forward propogation.

* [How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html): Unlike Colah's post, this long posts goes into detail explaining the math, notation, derivation of how backprop works in the context of Neural Networks. It's a really good read. To summarize, it:
	* First introduced the notation (and explain why they are set up that way)
	* Lay out the four fundamental equations for backprop:
		* An equation for the error in the output layer: δL
		* An equation for the error δl in terms of the error in the next layer, δl+1
		* An equation for the rate of change of the cost with respect to any bias in the network
		* An equation for the rate of change of the cost with respect to any weight in the network

 	* It also goes into proving the formulas 
 	* It then lays out the backprop algorithm
 	* It shows the code snippets for backprop
 	* It explains why it's much faster than forward propogation
 	* It then talks about a bit of the history of how backprop was discovered


* [Yes You Should Understand Backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b): A Medium post from Andre Karpathy explaining why every practitioners should learn about how backprop works. Essentially, his arguments is that even when doing DL in practice, if the activations are saturated, the gradient can become very close to 0 (vanishing gradient problem), causing the weight updates to be super slow, and in terms make the cost function to go down very slowly.


# Convolution Neural Network

* [CS231N CNN Intro](http://cs231n.github.io/convolutional-networks/): This post is, by far, the most comprehensive post that explains the architecture of CNN, the parameters involved, the input volume, the output volume, and more. It's a very meaty read, but gives a very detailed treatment of CNN. It focuses a lot on the architecture and different kind of layers in CNN, it doesn't talk much about how backpropogation works in the context of CNN. Some highlights:

	* A ConvNet architecture is in the simplest case a list of Layers that transform the image volume into an output volume (e.g. holding the class scores). 
		* There are a few distinct types of Layers (e.g. CONV/FC/RELU/POOL are by far the most popular). 
		Each Layer accepts an input 3D volume and transforms it to an output 3D volume through a differentiable function. 
		* Each Layer may or may not have parameters (e.g. CONV/FC do, RELU/POOL don’t)
		* Each Layer may or may not have additional hyperparameters (e.g. CONV/FC/POOL do, RELU doesn’t)

	* **Convolutional Layer (CONV)**: You need to have a good understanding of how input volume are transformed to output volume. 
		* The "neuron picture" that takes element-wise dot product of "input patch" on "fitler" + bias is very helpful
		* Know all the hyperparameters: depth, padding, stride, filter size. Know how to use these to calculate input & output volume dimensions
		* The numpy example + visualization demo is a good way to understnad how the math works

	* **Pooling Layer (P)**: 
		* Know the hyperparameters in pooling layer, and how to calculate dimensions

	* **Fully Connected Layers (FC)**
		* One can convert between CONV layer and FC layers

	* **ConvNet Architectures** is a great section: `INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

	* It also surveys different kind of CNN, quite useful overview of state of the art CNNs.

* [CNN Gently Explained on Medium](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721): A lot of visual explanations, focus on intuition.

* [Image kernels explained visually](http://setosa.io/ev/image-kernels/): Fantastic interactive visualizations of convolutions, and how we convolve image pixels with image kernels to get the output impages.

* Other materials from Colab, but less useful:

	* [Visualize CNN - from Colab](http://colah.github.io/posts/2014-07-Conv-Nets-Modular/): convolutional neural networks can be thought of as a kind of neural network that uses many identical copies of the same neuron.1 This allows the network to have lots of neurons and express computationally large models while keeping the number of actual parameters – the values describing how neurons behave – that need to be learned fairly small.

	* [Understand Convolution - from colab](http://colah.github.io/posts/2014-07-Understanding-Convolutions/): Understand the mathematical nature of convolution



