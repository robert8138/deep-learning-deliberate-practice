# Timeline

* **[Week of 2/18]: Warm-up** 
	* Surveyed materials, warm-up (fast.ai, CS 231N)
	* Created Github [Learning Project](https://github.com/robert8138/deep-learning-deliberate-practice)

* **[Week of 2/25]: Mathematical Formulation of Neural Nets**
	* Warm-up continued
	* Coursera DL course 1 (Basic of NN)
	* Studied backward propogation ([here](http://neuralnetworksanddeeplearning.com/chap2.html), [here](http://colah.github.io/posts/2015-08-Backprop/), and [here](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)).

* **[Week of 3/4]: Convolutional Neural Network**
	* Keras Book Chapter 1,2,4, and 5 (CNN) 
	* Finished Coursera DL course 2 (Optimization/Hyperparameter Tuning)
	* Study CNN in details (Stanford CS 231N [CovNet Notes](http://cs231n.github.io/convolutional-networks/) being the most useful)

* **[Week of 3/11]: Convolutional Neural Network (Interleaving)**
	* CS 231 coverage on Neural Network formulation
	* Finished Coursera DL course 4 (CNN)

* **[Week of 3/18]: Additional Topics in CNN / Hands-on Coding**
	* CS 231N (CNN Architecture, Visualizing CNN)
	* Deep Learning Software, Running Keras code on Redspot
	* Finished Coursera DL course 3 and started meta-topics on structuring ML projects

* **[Week of 3/25]: Hands-on Coding / Practical Advice for ML/DL**
	* Keras Book Chapter 7 on Functional API
	* Practical Tips for ML projects from Ian Goodfellow
	* Started and completed investigation about Image pipelines & CNN at Airbnb
	* Discovered Stanford CS 230
	* Finally read through [Rules of ML: Best Practice for ML Engineering](https://developers.google.com/machine-learning/rules-of-ml/)

* **[Week of 4/1]: Recurrent Neural Network & NLP**: I am a little bit hesitant to dig too deep in NLP just yet. It's a completely different topic and I still feel I haven't fully grasp CNN, so I will treat this as a quick survey for now. I think the proper way to learn it is to take [CS 224N](http://web.stanford.edu/class/cs224n/index.html).
	* Stanford CS 231N revisit Lecture 10 on RNN
	* Keras Book Chapter 6 on Recurrent Neural Network & RNN using Keras
	* Started Coursera DL course 5 - RNN
	* [CS 244N Word2Vec lecture notes](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf)

* **[Week of 4/8]: Recurrent Neural Network & NLP (Interleaving)**: to reinforce all the basic things I learned about word embeddings, RNNs from Coursera DL in Stanford class setting again
	* Stanford CS 224N Lecture 2: Word2Vec -> word embeddings
	* Stanford CS 224N Lecture 3: Co-ocurrence, GloVec word embeddings, BLEU scores
	* Stanford CS 224N Lecture 8: Vanilla RNN, Bidirectional RNN, Deep RNN
	* Stanford CS 224N Lecture 9: Gated Recurrent Units (GRU), Long-Short Term Memory (LSTM)
	* [Edwin Chen's explanation on LSTM with visualization](http://blog.echen.me/2014/05/30/exploring-lstms/)
	* [The Unreasonable Effectiveness of Recurrent Neural Network](http://karpathy.github.io/2015/05/21/rnn-effectiveness/): char RNN model -> {Paul Graham essays, Shakespeare, LaTex, C, Linux}

* **[Week of 4/15]: Tensorflow**
	* Stanford CS 224N: Lecture 4 (Backprop I), 5 (Backprop II), 13 (CNN) Review
	* Stanford CS 224N: Lecture 7 on Tensorflow
	* [Stanford CS 20](http://web.stanford.edu/class/cs20si/syllabus.html): Lecture 1-5, a more detailed introduction to Tensorflow
	* Glanced through Tensorflow's [Getting Started Guide](https://www.tensorflow.org/get_started/)
	* Scribed RNN notes in /concepts directory

* **[Week of 4/23]: Unsupervised Learning, Generative Models, Autoencoders**
	* Stanford CS 231 Lecture 13: Generative Model
	* Stanford CS 231 Lecture 16: Adversarial Examples and Adversarial Training
	* Keras Book Chapter 8: Generative Deep Learning
	* Keras Book Chapter 9: Conclusions (finished the book!)
	* [Generate Model from OpenAI](https://blog.openai.com/generative-models/)
	* Scribed CNN & CNN Architecture notes in /concepts directory

* **[Week of 4/30]: Review Week 1, summarize ideas in [/concepts directory](https://github.com/robert8138/deep-learning-deliberate-practice/tree/master/concepts)**
	* Stanford CS 229 Notes on DL, backprop (not as useful)
	* Scribe `learning_algorithms.md`, `learning_tricks.md`, and `learning_enhancement.md` 

* **[Week of 5/7]: Review Week 2, summarize ideas in [/concepts directory](https://github.com/robert8138/deep-learning-deliberate-practice/tree/master/concepts)**
	* Finished reading [Matrix Calculus for Deep Learning](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html), reminds me a lot of Math 105 - Multivariate Calculus & Measure Theory
	* Went through Michael Nielsen's [detailed treatment on backprop](http://neuralnetworksanddeeplearning.com/chap2.html) again, derived 4 fundamental equations of backprop along the way
	* My ugly derivation is written down [here](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/pictures/backprop_math_by_hand.png). This refreshes my Matrix Calculus quite a bit
	* Went through [Why are deep neural networks hard to train?](http://neuralnetworksanddeeplearning.com/chap5.html) and [Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) again, with the backprop derivatives in context
	* Scribed notes for Neural Network Formulation - basic structure, terminology, loss function
	* Scribe notes for Backprop - Revisit Gradient calculation for DL

# Upcoming

* **[Week of 5/14]: Google CapitalG's Machine Learning Training**
	* Machine Learning Training - advanced track in Google Cloud Campus
	* [Facebook's Field Guide to Machine Learning](https://research.fb.com/the-facebook-field-guide-to-machine-learning-video-series/)

* **[Week of 5/21]: Revisiting Convolutional Nueral Network (CS 20 materials + Work related projects)**
	* [Intro to ConvNet](https://docs.google.com/presentation/d/15E7NlyMkG8dAMa70i2OluprBDoz3UPyAk5ZpOiCkEqw/edit#slide=id.g1c60f09bdb_0_0)
	* ConvNet in Tensorflow: [Slides](https://docs.google.com/presentation/d/17VTArfQVtapBqfYecyvp3Kp9HKy8Pw2WI12acYME2nI/edit#slide=id.g1c60f09bdb_0_0), [Course Notes](https://docs.google.com/document/d/1ph43FB5fZ_iarPTjIXhdtDvHJOpk4ncI2vDyxnOWcqM/edit)
	* CS 231N Assignments: [Assignment 1](http://cs231n.github.io/assignments2017/assignment1/), [Assignment 2](http://cs231n.github.io/assignments2017/assignment2/), [Assignment 3](http://cs231n.github.io/assignments2017/assignment3/)