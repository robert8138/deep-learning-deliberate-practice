# Deep Learning Deliberate Practice

In 2018, I will take on a new learning project - to deliberately **learn** and **practice** Deep Learning.

## Some Inspirations

From Andrew Ng and his [answer](https://www.quora.com/How-should-you-start-a-career-in-Machine-Learning) on Quora:

> Every Saturday, you will have a choice between staying at home and reading research papers/implementing algorithms, vs. watching TV. If you spend all Saturday working, there probably won't be any short-term reward, and your current boss won't even know or say "nice work." Also, after that Saturday of hard work, you're not actually that much better at machine learning. But here's the secret: If you do this not just for one weekend, but instead study consistently for a year, then you will become very good.

Anecdotal story: Aside from taking his CS 229 at Stanford when I was a student there, I've ran into him several times at different cafes around South Bay on Saturdays, and everytime I saw him, he was either working or learning something, very inspiring!

## Recap on Previous Learning Projects

In early 2016, I made the [decision](https://medium.com/@rchang/advice-for-new-and-junior-data-scientists-2ab02396cf5b) to transition from Type A Data Scientist to Type B Data scientist. As I started taking on less inference work and more data engineering and machine learning projects. I took on a [leanring project](https://github.com/robert8138/python-deliberate-practice) in late 2016 to switch from R to Python. This was a "pre-requisite" for me to succeed at Airbnb because doing Data Engineering or building Machine Learning models at Airbnb are generally much easier with Python. This small learning project turned out to be one of the best investments that I made for my technical growth at Airbnb.

Because of this investment, I was able to march on and take on two very big projects ([Listing LTV](https://medium.com/airbnb-engineering/using-machine-learning-to-predict-value-of-homes-on-airbnb-9272d3d4739d) and Homes Metrics), both of which heavily involved Airflow and frameworks built on top of Airflow. Specifically, I was able to navigate the codebase of these frameworks, find bugs, and become a better user and collaborator with DE and ML Infra. Throughout the process, I learned a ton about how to build offline training, offline scoring ML models, and my knowledge in Data Engineering has increased dramatically. 

Interestingly, 2017 was the first year in my professional career where the majority of my technical growth came from on the job training rather than self study. This was one big experiment, but I think it paid off. This approach was also aligned with what Cal Newport and Scott Young advocate:

> The best approach wasn’t to try to persuade people to carve out huge chunks of their life for deliberate practice (which they were unable to do), but rather to carefully pick projects that would be able to simultaneously reach both goals: achieving a legitimate object of work and also pushing deliberate practice.

## Motivation for Learning DL

In early 2018, I have decided to transition to the Select team to focus on applied Machine Learning. Specifically, there is general buy-ins from the team, my future manager, and my immediate teammates that Machine Learning will be critical to scale the program in 2018. Among of the many areas that we can tackle on Select, one of which is to better leverage Airbnb's collection of Home images. 

As a result, I foresee that the class of problems that I will be dealing with, fall closely under the realm of Deep Learning, and this is the best opportunity for me to achieve deliberate practice on an increasingly important skill that will allow us to unlock enormous business values in the future.

## Course Planning

Like any learning project, it is generally wise to take a week or two to compile the list of study materials, trial study them, and then finalize on the final curriculum. Luckily, this topic is so hot that there are already many useful obvious candidates or even reviews of these candidate courses. Here are a few:

* [Thoughts after taking the Deeplearning.ai courses](https://towardsdatascience.com/thoughts-after-taking-the-deeplearning-ai-courses-8568f132153): A good comparison of Fast.ai course and Coursera series.

### University Courses

* [Fast.ai](http://wiki.fast.ai/index.php/Main_Page): Top-down teaching approach by Jeremy Howard and Rachel Thomas
* [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning): Bottom-up teaching approach
* [Stanford CS 231N: Convolutional Neural Network](http://cs231n.stanford.edu/syllabus.html): This is highly recommended by Jiaying and Heerad. This is probably also the most rigorous course of the three mentioned here because it's an actual Stanford Course. I was told the Homeworks are superb and I should definitely do them.
* [Stanford CS 224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/): Focus on NLP, word vector representations, window-based neural networks, recurrent neural networks, long-short-term-memory models, recursive neural networks, convolutional neural networks as well as some recent models involving a memory component.
* [Stanford CS 230: Deep Learning](http://cs230.stanford.edu/syllabus.html): This course is based on Andrew Ng's Coursera deep learning specialization courses, with additional materials taught during classroom settings. Great classroom notes and [Tutorials on Tensorflow](https://cs230-stanford.github.io/).

### Toolings: Keras & Tensorflow

* [Deep Learning in Python](https://www.datacamp.com/courses/deep-learning-in-python?tap_a=5644-dce66f&tap_s=93618-a68c98): Deep learning is the machine learning technique behind the most exciting capabilities in diverse areas like robotics, natural language processing, image recognition and artificial intelligence (including the famous AlphaGo). In this course, you'll gain hands-on, practical knowledge of how to use deep learning with Keras 2.0, the latest version of a cutting edge library for deep learning in Python.

* [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python): Deep Learning with Python introduces the field of deep learning using the Python language and the powerful Keras library. Written by Keras creator and Google AI researcher François Chollet, this book builds your understanding through intuitive explanations and practical examples. I heard a lot of great things about this book.

* [Stanford CS 20: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/): This course will cover the fundamentals and contemporary usage of the Tensorflow library for deep learning research. We aim to help students understand the graphical computational model of TensorFlow, explore the functions it has to offer, and learn how to build and structure models best suited for a deep learning project. Through the course, students will use TensorFlow to build models of different complexity, from simple linear/logistic regression to convolutional neural network and recurrent neural networks to solve tasks such as word embedding, translation, optical character recognition, reinforcement learning. Students will also learn best practices to structure a model and manage research experiments.

* [Coursera/Google Cloud ML Specialization](https://www.coursera.org/learn/intro-tensorflow): Introduction to Tensorflow

### Textbook

* [Deep Learning](http://www.deeplearningbook.org/): Written by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This is the book which covers in detail the mathematical underpinnings of many DL topics.

### Blogs 

#### Deep Learning

* [Colab](http://colah.github.io/): His intuitive explanation of DL topics are very well regarded, and something that I plan to revisit from time to time.
* [Andrej Karpathy's blog](http://karpathy.github.io/)
	* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
	* [Yes, you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)

#### Machine Learning in General

* **Practical Machine Learning Advice**
	* [Andrew Ng's ML Yearning - WIP](http://www.mlyearning.org/): This book is all about practical advice for how to tackle challenges in ML projects. He started off this project but got sidetracked by the Coursera DL specialization course. He is coming back to it now.
	* [Rules of ML: Best Practice for ML Engineering](https://developers.google.com/machine-learning/rules-of-ml/): This document is intended to help those with a basic knowledge of machine learning get the benefit of best practices in machine learning from around Google. [PDF]((http://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)) is also available.
	* [Machine Learning: The High-Interest Credit Card of Technical Debt](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43146.pdf): The goal of this paper is highlight several machine learning specific risk factors and design patterns to be avoided or refactored where possible. These include boundary erosion, entanglement, hidden feedback loops, undeclared consumers, data dependencies, changes in the external world, and a variety of system-level anti-patterns.
	* [The Two Cultures of Machine Learning Systems](https://medium.com/opendoor-labs/the-two-cultures-of-machine-learning-systems-c648db0bb4d8): How ML Engineers and Data Scientists approach ML differently
* [Dimensions of Machine Learning Models](https://www.slideshare.net/SharathRao6/lessons-from-integrating-machine-learning-models-into-data-products): Less relevant to Deep Learning specifically, but this is still by far the best piece that I can find in categorizing different ML models (contextual data dimension v.s. latency requirement dimension). [Video](https://www.youtube.com/watch?v=wG5EyHYrJGE&t=891s) from DataEngConf2017 is also available.


### Beyond Self-Learning

* **Research Track**: 
	- Try to implement basic neural network from scratch (e.g. CS 231N homeworks)
	- Read a lot of papers from [arxiv](https://arxiv.org/), then try to replicate the results

* **Application Track**:
	- Do it the Jeremy Howard / hacker way, try to map techniques to problems, or problems to techniques
	- A lot of opportunity to do this on my current team, since we have a lot of image-related tasks