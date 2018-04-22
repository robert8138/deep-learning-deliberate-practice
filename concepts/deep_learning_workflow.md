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