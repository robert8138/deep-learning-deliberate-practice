# [TODO] Formulation for Training Neural Network

* Defining the loss function = error + regularization
* Defining the neural net function, parameters to be trained, hyperparameters to be tuned
* Defining the optimization problem w.r.t parameters to be trained


# [TODO] Backward Propogation

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