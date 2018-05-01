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