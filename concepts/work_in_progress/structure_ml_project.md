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