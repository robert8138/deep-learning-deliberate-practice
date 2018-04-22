# Deep Learning Workflow

* **Defining the problem and assembling a dataset**
	* What will your input data X be? What are you trying to predict (y)?
	* What type of problem are you facing? Is it binary classification? Multiclass classification? Scalar regression? Vector regression? Multiclass, multilabel classification? Something else, like clustering, generation, or reinforcement learning? Identifying the problem type will guide your choice of model architecture, loss function, and so on.
	* Be aware of the hypotheses you make at this stage:
		* You hypothesize that your outputs can be predicted given your inputs
		* You hypothesize that your available data is sufficiently informative to learn the relationship between inputs and outputs

* **Choosing a measure of success**
	* To achieve success, you must define what you mean by success—accuracy? Precision and recall? Customer-retention rate? Your metric for success will guide the choice of a loss function.
	* For common loss functions for different tasks, see [CS 231's Neural Network Notes II](http://cs231n.github.io/neural-networks-2/#losses).
	* Guideline from Ian Goodfellow in his [book](http://www.deeplearningbook.org/slides/11_practical.pdf):
		* Accuracy? (High accuracy v.s. low accuracy requirement)
		* Coverage? (% of examples processed)
		* Precision? (% of predictions that are right)
		* Recall? (% of objects detected)
		* Amount of error? (For regression problems)

* **Deciding on an evaluation protocol**
	* Maintaining a hold-out validation set: The way to go when you have plenty of data. The the modern-era of DL and big data, we typically do not do 70(train)/30(test) or 60(train)/20(validation)/20(test) anymore. It's more like 98/1/1 if you have a large dataset.
	* Doing K-fold cross-validation: The right choice when you have too few samples for hold-out validation to be reliable
	* Doing iterated K-fold validation: For performing highly accurate model evaluation when little data is available

* **Data pre-processing**
	* Vectorization (reshape them into tensors)
	* Value normalization (between 0 and 1), for normalizing later layers' activation, use batch normalization (technically learned, not data pre-processing).
	* Handle Missing values

* **Developing a model that does better than a baseline**
	* Get up and running ASAP
	* Build the simplest viable system first
	* Use the model that you know (decide shallow or deep)
	* When to use Deep
		* No structure -> fully connected: 2-3 hidden layers, ReLu, BatchNorm, Adam
		* Spatial structure -> convolutional: download a pretrained model, BatchNorm, Adam
		* Sequential structure -> recurrent: LSTM/SGD, Gradient clipping, high forget gate bias
	* If using deep learning 
		* Decide what is the last layer configuration
		* Optimizatino routine (Gradient Descent, Momentum, RMSprop, ADAM ... etc)

* **Data-driven Adaptation**
	* Choose what to do based on data: Inspect data for defects (least confident predictions)
	* Plot training and dev set error to investigate under- v.s. over-fitting.

* **Develop a model that overfits**
	* Add layers
	* Make the layers bigger
	* Train for more epochs

* **Regularizing your model**
	* Data Augmentation
	* Add dropouts
	* Add L1/L2 regularization
	* Add or remove layers

* **Tune hyper-parameters**
	* Prefer random search over grid search
	* Reason about how changing hyperparameter would change model capacity/complexity

* **Final model**
	* Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it one last time on the test set.



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




# Structuring Machine Learning Projects

* [**How to think about problem formulation**](http://cs230.stanford.edu/files/Week2_slides.pdf)
	- Input: What is the the input of the model, number of inputs, size of input
	- Output: What is the output of the model, number of outputs
	- Data: What data do you have?
	- Architecture: Shallow network, Deep network (standard or customized)
	- Loss: How would you optimize the loss of your learning problems
	- Training: What is the training process learning? parameters / input images?

* **Meta ML Project Strategy - Orthogonalization**: A good performing ML model usually assumes the following assumption (or need to satisfy the following conditions):
	- It has a good **cost function** capturing reality
		- Here, the art really is think about how to formulate the model and the associated cost
	- Fit **training set** well on the cost function (i.e. reduce model bias):
		- Fit a bigger network
		- Use better optimization routine
		- Train the model longer
		- Make the model more complex by trying different architectures
	- Fit **dev set** well on the cost function (i.e. reduce model complexity/variance)
		- L1, L2 Regularization
		- Dropout
		- collect more data
	- Fit **test set** well on the cost function
		- always make sure dev and test set have the same distribution
		- collect more data on the dev set if test error >> dev error (maybe the dev set is not representative enough)
	- Performs well in the real world
		- If does well all the way to test set, change the dev/test set to fit reality or make the cost function closer to reality

* **Setting up Cost Function**
	- Loss functions for standard problems are described in [CS 231N notes](http://cs231n.github.io/neural-networks-2/#losses)
	- For more complex problems, you might have to come up with your own loss function (e.g. Airbnb's smart pricing problem has its unique loss function - booking regrets)
	- Sometimes we might need to change the loss function. For example, if we learned that our algorithm is consistently misclassifying certain examples, maybe we should put more weights on them (e.g. porn pictures). 

* **Come up with Evaluation Metrics**
	- Choose a single evaluation metric is very useful so we don't compare models with different metrics (e.g. precision & recall -> F score)
	- If there are multiple metrics under consideration, frame it as one optimizing metric & one or more satisficing metric. If framed as an optimization problem, optimizing metric is the thing that we are trying to maximize in the objective function, while satisficing metric are constraints we would like to satisfy.

* **Splitting Data into Train/Dev/Test Sets**
	- **Size**: 
		- In the traditional ML world with smallish data, we might do 60%/20%/20% split. With DL big data era, we might do 98%/1%/1%, where even with 1%, dev and test set are big enough
	- **Distribution**: 
		- Make sure that dev/test set are sampled exactly the same way (i.e. distribution of dev ~ distribution of test).
		- Choose a dev and test set to reflect data you expect to get in the future (i.e. distribution dev/test ~ distribution of real world)
		- Think of the analogy of Bullseye & Arrows. The location of the Bullseye is the distribution where our dev/test should be (they should be close to future unseen data). The metric is our arrow that allows to see how close we are.
		- There are going to be times where distribution of training set differs from dev/test set. That's actually acceptable as long as the trained model does well on dev/test set! (Transfer learning is a good example of this)

* **Understand Model Performance**
	- Typically, we would track the following error rate for model performance. To see an overview, see this [coursera lecture](https://www.coursera.org/learn/machine-learning-projects/lecture/ht85t/bias-and-variance-with-mismatched-data-distributions)
		- Human-level Performance
		- Training set error
		- Training-dev set error
		- Dev set error
		- Test set error
	- **META POINT**: For DL, a lot of the tasks are perception related tasks rather than ML on structured data. The big difference between these ML projects is that humans are generally really good at perception-based tasks, and can get to the Bayes optimal error, whereas humans are much worse at ML on structured data (online advertising, product recommendation ... etc)
	- For perception-based tasks, human-level performance is generally a good **proxy** to **Bayes Optimal error**. This is important because we can use this as a benchmark to understand how far our training set error is away from bayes error. The gap between **Bayes error ~ human-level performance** and **training error** gives us an idea on how much **avoidable bias** we can still improve. This corresponds to the **bias** in the bias and variance trade-off.
	- Typically, the gap(training error, dev error) tells us the variance of the model, meaning how well the model is able to generalize to data with more variant pattern. This is the **variance** in the bias and variance trade-off.
		- However, sometimes the training data distribution will differ from the dev set distribution, so if we see a gap in training and dev set error, two things are influx - generalization ability to a fix pattern & new patterns emerges. To combat this, we will use **training-dev set error** - this data is a subset of the training data, but will not be used for training.
			- Gap(training error, training-dev error) tells us how well the trained model generalize to patterns in one distribution. This is similar to gap(training, dev) when they come from the same distribution
			- Gap(training-dev error, dev set error) tells us the extent in which training & dev set distribution differs.
		- To address data distribution mismatch, one way to do so is to make training data similar to dev/test set, another way is to analyze how training and dev set differs, and address them specifically. There is currently no agreeable standard.

* **Error Analysis**
	- With the model performance breakdown set up, we can do error analysis to understand why the model isn't doing well (Andrew still do this for his projects, and put them in a spreadsheet)
		- Sample the examples where we made a mistake
		- Tally up the reasons of why we are making mistakes
		- Tackle the reason that has the largest share of mistakes
		- If there are incorrectly labeled examples, that can be another category as well
	- It's always better to set up the system, try something quick and dirty, then start doing error analysis and iterate

* **Other Topics: Transfer Learning**
	- In DL era, it's very popular to leverage pre-trained model to fine tune adjacent problems. When should one use transfer learning from task A -> B?
		- Task A and B have the same input X
		- You have a lot of data from Task A than Task B
		- Low level concept from A could be helpful for learning task B
	- There are different ways to do transfer learning
		- You can simply remove the last layer of a pre-trained model, replace with a new layer with randomly initialized weights for that final layer, then use task B's {x,y} to fine tune the parameters.
		- If you have enough data for task B, you can even go back a few layers, and then retrain the parameters in the last few layers
		- The final extreme case is to use the pretrained model's weight as weight initialization, then retrain the entire model, so the pre-trained model only serve as weight initialization
		- For all the options above, you can choose to add and extend more layers if you like.

* **Other Topics: Multi-task Learning**
	- Not as popular as transfer learning, but quite useful for things like object detection, where you ended up having a vector of y, and you encapsulate several tasks in one big y vector, and learn in one network.
	- When does Multi-task Learning makes sense:
		- Training on a set of tasks that could benefit from have shared features
		- Amount of data for each task is quite similar (so can combine forces)
		- Have neough computational power to train oen big neural network together

* **Other Topics: End-to-End Learning**
	- I called this as when should I decompose my problems into smaller problem
	- End-to-End learning means to learn a direct mapping of X -> Y, this is generally really hard.
	- The trade-off really is in how much data you have, if you have a lot of data, end-to-end learning is more likely, but in the absence of data, having more hand-designed features makes more sense.


