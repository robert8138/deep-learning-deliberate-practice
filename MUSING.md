# History of Machine Learning

Francis Chollet's exposition on the history of Machine Learning in his book 'Deep Learning with Python' is really good. I really appreciate his explanation because I feel like he is narrating the history of ML for me, and it allows me to see how my academic training fits into just a very small sub-segment of this history.

From simple logistic regression/Naive Bayes classifier, early nueral nets, Kernel methods/SVM, to tree models, he essentially explains all the things that I was taught when I was in school up until 2012, and all of these trainings are around shallow-learning - methods that learn representation of data that are 1 or 2 layers.

# Machine Learning as an Engineering Science

In general, three technical forces are driving advances in machine learning:

* Hardware
* Datasets and benchmarks
* Algorithmic advances

Because the field is guided by experimental findings rather than by theory, algorithmic advances only become possible when appropriate data and hardware are available to try new ideas (or scale up old ideas, as is often the case). Machine learning isn’t mathematics or physics, where major advances can be done with a pen and a piece of paper. It’s an engineering science.

# Deep Learning

For Deep Learning, the resurgence is driven by all three forces:

* Hardware: GPU adoption and adaptation to DL training
* Datasets and benchmarks: ImageNet as an example, Kaggle competition
* Algorithm advances: better _activation functions_, better _weight initialization schemes_, better _optimization routines_ ... etc. It's interesting that it's less about new learning algorithm, it's still centered around neural nets.

# Shallow Learning v.s. Deep Learning

Almost all the things that I learned in school are shallow-learning (even kernel method for SVM is one layer). The state of the art methods like Gradient Boosted Trees are also shallow methods (no data transformation, just cut the space with the original representation). 

The fundamental breakthrough of Deep Learning is that:


```
the incremental, layer-by-layer way in which increasingly complex representations are developed,
and the fact that these intermediate incremental representations are learned jointly, each layer
being updated to follow both the representational needs of the layer above and the
needs of the layer below. 
```

# The three phases of AI developement

* 1950s/1960s: Marvin Minsky, chess game, and Symbolic AI. A lot of hypes, but eventually didn't live up to the prediction of human-level general intelligence within 10 years.
* 1990s: Expert systems, progress backed by large corporations that eventually also died.
* Early 2010s: Resurgence of Deep Learning, data, hardware, algorithm development. 

# Historical Fun Facts

* Geoffrey Hinton discovered backward propagation in the 1980s
* Yecunn developed LeNet which are used by USPS to recognized digits in the 1990s
* GPUs, originally developed for Gaming industry/application, became the hardware innovation for DL
* Hinton and his group start making breakthrough for ImageNet competition in 2012s
* Google dedicated to developed specialized chips, TPU, for DL training
* George Hinton introduced RMSprop in his Coursera course, and everyone cites the Coursera reference in their papers
* Inception Network was inspired by the Meme of Movie Inception

# Thoughts from Andrew's DL specialization course 3 on structuring ML projects on Plus ML Projects

* Lead Scoring Model
	- We do not have the right loss function (It should have been cross-entropy loss instead of mean squared loss)
	- We do not have clear performance tracking for training, dev, test error
	- We should have misclassification error as evaluation metric, not just a lift chart
	- Distribution of training data != Distribution of dev/test set
		- We have a situation where there is feedback loop, so distribution of training data != dev/test data, so need to make training data as closely as possible to dev set
		- Our dev/test set is not quite aligned with what the real world is going to be (because we are launching in new markets)
	- If the model is not doing well, we should do error analysis to understand what's going on
	- This is fundamentally a perception-based problem, so we should move away from structure problems. Furthermore, we should break it down into specific image classification problems using CNN, so to avoid this psuedo end-to-end approach - why do we think using revenue, price point, market, and availability will help us to pick Select like listings, we are simply banking on correlation between quality & performance, but it's not enough.