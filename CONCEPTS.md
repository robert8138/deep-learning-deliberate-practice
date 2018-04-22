# [TODO] Formulation for Training Neural Network

* Defining the loss function = error + regularization
* Defining the neural net function, parameters to be trained, hyperparameters to be tuned
* Defining the optimization problem w.r.t parameters to be trained



# [TODO] Data Augmentation / Transfer Learning

* Data Augmentation: Enlarge your training data
	* Particularly helpful for image classification problems, when one introduces perturbation to the data to make it richer

* Transfer Learning: Don't start from scratch
	* If you have very little data, take trained neural network as is, pop the very last softmax layer while freezing the weights on all prior layers, and impose your own softmax layer with your problem specific labels.
		- Appropriate to use when the new training data is small
		- Good if the trained labels are similar to the new task labels, otherwise might not work well
		- Often useful to precompute the embeddings and save them to disk, then train on new labels

	* Take trained neural network as is, pop the very last **FEW** layers, and retrain the last few layers with the new labels
		- Appropriate to use when you have a larger label dataset
		- By retraining, we can either retrain the last few layers with the same architecture
		- or we can retrain them on with our own architecture for the last few layers

	* Use the same architecture as the pre-trained model, but retrain the entire network
		- Appropriate to use when you have a lot of data
		- Use the pre-trained weights as your initialization weight (and replace the original random initialization), then retrain everything
		- It takes time and computational power/budget




# Optimization

Andrew's DL specialization, especially [Course 2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome), in my opinion, does the best job in explaining the topics of optimization and graident descient. 

After learning the basics about Neural Network, the parameters involved, and the loss function formulation, the main concept is then to minimize the loss function, given the input parameters. This is fundamentally a optimization problem. Because the functional form of a Nueral Network can be unwieldy, numerical optimizations are the best way to approach this.

Of all the numerical approaches, *gradient descent* is one of the most fundamental approaches, but many additional ideas have been developed, such as **momentum**, **RMSProp**, **ADAM**, all aim to speed up the process for the graident to reach the minimum. 

* [**Understand Gradient Descent**](http://cs231n.github.io/optimization-1/): By now, I kind of take this for granted already. But Stanford's CS 231N did a good job in explaining how we can approach optimization (random search, random local search, or follow the graident). It can be shown, mathematically, that moving in the direction of the gradient, is the most efficient.

* **How Far Should I move my gradient?** _The choice of how many training examples to process to update the gradient can be a very important choice_.

	* **Batch Gradient Descent**: This is known as taking one complete pass of the training data TOGETHER, and calculating the gradient over the entire training data, and update gradient once. We can do this efficiently using vectorization, but sometimes we can overshoot because the gradient takes a large step.

	* **Stochastic Gradient Descent**: The other extreme, where we take one example as ONE pass, so we are only calculating the gradient over one training example. The advantage of this is that we can move by a mini-step, but then we lose all the advantages of vectorization.

	* **Mini-batch Gradient Descent**: This is the preferred choice, where we take mini-batches of the training data, so we do not take a complete pass of the entire training data. Instead, we take small batches of it, so it might take several rounds before we take one pass (i.e. one epoch). The advantage is that we can average out the gradient direction where it has high variance, while enjoying vectorization.

* **More Sophisticated Tweaks on Top of Gradient Descent**: _DL researchers have been doing a lot of work to speed up the optimization routines, but it turns out that a lot of them do not generalized well, the ones that do generalized well involved calculating the **expoential weighted averages** of the first and second moment of the gradients._

	* [**Exponential Weighted Average**](https://www.coursera.org/learn/deep-neural-network/lecture/Ud7t0/understanding-exponentially-weighted-averages): I love Andrew's explanation on this topic, it's the math behind these newer optimization routines. The idea is that we will take the exponential weighted average of the first & second moment of the gradient, to smooth up the updates.

	* [**Gradient Descent Momentum**](https://www.coursera.org/learn/deep-neural-network/lecture/y0m1f/gradient-descent-with-momentum): The key idea is to smooth out the gradient using exponential weighted averages, to make the gradient less brittle. Instead of updating by learning_rate * gradient, we update `learning_rate * exponential_weighted_avg(gradient)`

	* [**Gradient Descent with RMSprop**](https://www.coursera.org/learn/deep-neural-network/lecture/BhJlm/rmsprop): The key idea is to update less aggressively on gradient directions that are volatile, and update more aggressively on gradient directions that are stable. Instead of updating by learning_rate * gradient, we update `learning_rate * gradient / exponential_weighted_avg(gradient ** 2)`

	* [**ADAM**](https://www.coursera.org/learn/deep-neural-network/lecture/w9VCZ/adam-optimization-algorithm): This combines Momentum & RMSprop, so not only do we average out the brittle gradient direction, we update more aggressively on gradient directions that are stable, and less aggressively on gradient directions that are volatile. Update by `learning rate * exponential_weighted_avg(gradient) / exponential_weighted_avg(gradient ** 2)`
* **Other Important Topics for Optimization**
	* **Initialization**: 
		* Normalize the weights: this is to standardize the weights, so to standardize the output, so to standardize the gradient, so we don't run into vanishing/exploding gradient. **Vanishing graident** is bad because it means your learning algorithm is not updating, and **exploding gradient** is bad because it means you are taking too big of a step, and overshooting.
	* **Gradient Checking**: this is more relevant if you are building your own optimization routine, where you check the gradient calculation from the analytical differentiation is roughly the same as the numerical derivative.
	* **Batch Normalization**: To make sure that inputs in each layer are somewhat gaussian. The idea is to normalize it, and give it the chance to stay gaussian, or go back to its original form. Usually they stay somewhat guassian, which improves stability in gradient descent. Andrew's explanation from Coursera DL course 2 is the best that I have seen.



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



# Regularization

In traditional machine learning, **Bias and Variance trade-off** is a critical concept to understand, because depending on whether the model is suffering from a bias problem or a variance problem, the approach in improving the model performance would be quite different. In the era of DL, the trade-off still exist, but are not as big. Therefore, the workflow for DL typically becomes 1). overfit as much as you can, to reduce the bias, and to lower the training error, then 2). try adding more data (e.g. data augmentation), or applying regularization, to reduce variance, to bring down validation/test error.

The theme of regularization is to reduce model complexity, and model complexity is roughly captured by the "magnitudes", "norm", "number" of the weights. There are different ways to regularize the model.

* **L1/L2 regularization**: 
	- Intuition 1: this approach limits the magnitudes or norm of the parameters being trained.  
	- Intuition 2: by making the magnitudes of the parameters small, the activation function act more like an activation function (for tahn or sigmoid activation functions)
* **Dropout**: this approach directly limit the number of hidden units in each hidden layer, reducing the complexity of the model
* **Early Stopping**: this approach stops training before the model starts to overfit. It's sometimes used but Andrew doesn't recommend it, because it doesn't clearly separate the tasks of overfit as much as you can, then regularize. 



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
