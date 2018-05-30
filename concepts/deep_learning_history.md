# History of Machine Learning

Francis Chollet's exposition on the history of Machine Learning in his book 'Deep Learning with Python' is really good. I really appreciate his explanation because I feel like he is narrating the history of ML for me, and it allows me to see how my academic training fits into just a very small sub-segment of this history. The notes below are scribed from his book, all copyright belongs to the author.

Table of Content

* [Nomenclature](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#nomenclature)
	* [Artificial Intelligence](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#artificial-intelligence)
	* [Machine Learning](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#machine-learning)
		* [Learning Representation from Data](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#learning-representation-from-data)
	* [Deep Learning](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#deep-learning)

* [What Makes Deep Learning Different](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#what-makes-deep-learning-different)

* [History of AI Hype Cycles](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#history-of-ai-hype-cycles)
	* [Recent Breakthroughs](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#recent-breakthroughs)
	* [What's different this time](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#whats-different-this-time)

* [The Modern Machin Learning Landscape](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#the-modern-machin-learning-landscape)

* [Takeaways](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#takeaways)

* [Historical Fun Facts](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/deep_learning_history.md#historical-fun-facts)

## Nomenclature

### Artificial Intelligence

Artificial intelligence was born in the 1950s, when a handful of pioneers from the nascent field of computer science started asking whether computers could be made to “think”—a question whose ramifications we’re still exploring today. A concise defini- tion of the field would be as follows: the effort to automate intellectual tasks normally per- formed by humans. As such, AI is a general field that encompasses machine learning and deep learning, but that also includes many more approaches that don’t involve any learning.

For a fairly long time, many experts believed that human-level artificial intelligence could be achieved by having programmers handcraft a sufficiently large set of explicit rules for manipulating knowledge. This approach is known as symbolic AI, and it was the dominant paradigm in AI from the 1950s to the late 1980s. It reached its peak popularity during the expert systems boom of the 1980s.

Although symbolic AI proved suitable to solve well-defined, logical problems, such as playing chess, it turned out to be intractable to figure out explicit rules for solving more complex, fuzzy problems, such as image classification, speech recognition, and lan- guage translation. A new approach arose to take symbolic AI’s place: machine learning.

### Machine Learning

Machine learning arises from this question: could a computer go beyond “what we know how to order it to perform” and learn on its own how to perform a specified task? Could a computer surprise us? Rather than programmers crafting data-processing rules by hand, could a computer automatically learn these rules by looking at data? This question opens the door to a new programming paradigm.

A machine-learning system is trained rather than explicitly programmed. It’s presented with many examples relevant to a task, and it finds statistical structure in these exam- ples that eventually allows the system to come up with rules for automating the task.

Although machine learning only started to flourish in the 1990s, it has quickly become the most popular and most successful subfield of AI, a trend driven by the availability of faster hardware and larger datasets.

#### Learning Representation from Data

A central problem in machine learning and deep learning is to **meaningfully transform data**: in other words, to learn useful **representations** of the input data at hand—representations that get us closer to the expected output. Before we go any further: what’s a representation? At its core, it’s a different way to look at data to represent or encode data.

**Machine-learning models are all about finding appropriate representations for their input data—transformations of the data that make it more amena- ble to the task at hand, such as a classification task**. Learning, in the context of machine learning, describes an automatic search process for better representations.

So that’s what machine learning is, technically: searching for useful representa- tions of some input data, within a predefined space of possibilities, using guidance from a feedback signal. This simple idea allows for solving a remarkably broad range of intellectual tasks, from speech recognition to autonomous car driving.

### Deep Learning

Deep learning is a specific subfield of machine learning: a new take on learning repre- sentations from data that puts an emphasis on learning successive layers of increasingly meaningful representations. The deep in deep learning isn’t a reference to any kind of deeper understanding achieved by the approach; rather, it stands for this idea of successive layers of representations. You can think of a deep network as a multistage information-distillation operation, where information goes through successive filters and comes out increasingly purified.

Other appropriate names for the field could have been layered representations learning and hierarchical representations learning. Modern deep learning often involves tens or even hundreds of successive layers of representations — and they’re all learned automatically from exposure to training data. These layer representations are (almost always) learned via models called **nueral networks**.

Meanwhile, other approaches to machine learning tend to focus on learning only one or two layers of representations of the data; hence, they’re sometimes called shallow learning. These are most of the algorithms that I was taught in school.

## What Makes Deep Learning Different

The primary reason deep learning took off so quickly is that it offered better performance on many problems. But that’s not the only reason. Deep learning also makes problem-solving much easier, because **it completely automates what used to be the most crucial step in a machine learning workflow: feature engineering.**

Previous machine-learning techniques—shallow learning—only involved trans- forming the input data into one or two successive representation spaces, usually via simple transformations such as high-dimensional non-linear projections (SVMs) or decision trees. But the refined representations required by complex problems gener- ally can’t be attained by such techniques. As such, humans had to go to great lengths to make the initial input data more amenable to processing by these methods: they had to manually engineer good layers of representations for their data. This is called feature engineering. Deep learning, on the other hand, completely automates this step: with deep learning, you learn all features in one pass rather than having to engineer them yourself. This has greatly simplified machine-learning workflows, often replac- ing sophisticated multistage pipelines with a single, simple, end-to-end deep-learning model.

You may ask, if the crux of the issue is to have multiple successive layers of representations, could shallow methods be applied repeatedly to emulate the effects of deep learning? In practice, there are fast-diminishing returns to successive applications of shallow-learning methods, because the optimal first representation layer in a three- layer model isn’t the optimal first layer in a one-layer or two-layer model. What is transformative about deep learning is that it allows a model to learn all layers of representation jointly, at the same time, rather than in succession (greedily, as it’s called). With joint feature learning, whenever the model adjusts one of its internal features, all other features that depend on it automatically adapt to the change, without requiring human intervention.

Everything is supervised by a single feedback signal: every change in the model serves the end goal. This is much more powerful than greedily stacking shallow models, because it allows for complex, abstract representations to be learned by breaking them down into long series of intermediate spaces (layers); each space is only a simple transformation away from the previous one.

These are the two essential characteristics of how deep learning learns from data: the _incremental, layer-by-layer way in which increasingly complex representations are developed_, and the fact that _these intermediate incremental representations are learned jointly_, each layer being updated to follow both the representational needs of the layer above and the needs of the layer below. Together, these two properties have made deep learning vastly more successful than previous approaches to machine learning.

## History of AI Hype Cycles

Twice in the past, AI went through a cycle of intense optimism followed by disappointment and skepticism, with a dearth of funding as a result. It started with symbolic AI in the 1960s. In those early days, projections about AI were flying high. One of the best-known pioneers and proponents of the symbolic AI approach was Marvin Minsky, who claimed in 1967, “Within a generation ... the prob- lem of creating ‘artificial intelligence’ will substantially be solved.” Three years later, in 1970, he made a more precisely quantified prediction: “In from three to eight years we will have a machine with the general intelligence of an average human being.” In 2016, such an achievement still appears to be far in the future—so far that we have no way to predict how long it will take—but in the 1960s and early 1970s, several experts believed it to be right around the corner (as do many people today). A few years later, as these high expectations failed to materialize, researchers and government funds turned away from the field, marking the start of the first AI winter (a reference to a nuclear win- ter, because this was shortly after the height of the Cold War).

It wouldn’t be the last one. In the 1980s, a new take on symbolic AI, expert systems, started gathering steam among large companies. A few initial success stories triggered a wave of investment, with corporations around the world starting their own in-house AI departments to develop expert systems. Around 1985, companies were spending over $1 billion each year on the technology; but by the early 1990s, these systems had proven expensive to maintain, difficult to scale, and limited in scope, and interest died down. Thus began the second AI winter.

We may be currently witnessing the third cycle of AI hype and disappointment— and we’re still in the phase of intense optimism. It’s best to moderate our expectations for the short term and make sure people less familiar with the technical side of the field have a clear idea of what deep learning can and can’t deliver.

### Recent Breakthroughs

Around 2010, although neural networks were almost completely shunned by the scientific community at large, a number of people still working on neural networks started to make important breakthroughs: the groups of Geoffrey Hinton at the Uni- versity of Toronto, Yoshua Bengio at the University of Montreal, Yann LeCun at New York University, and IDSIA in Switzerland.

In 2011, Dan Ciresan from IDSIA began to win academic image-classification competitions with GPU-trained deep neural networks — the first practical success of modern deep learning. But the watershed moment came in 2012, with the entry of Hinton’s group in the yearly large-scale image-classification challenge ImageNet. The ImageNet challenge was notoriously difficult at the time, consisting of classifying high- resolution color images into 1,000 different categories after training on 1.4 million images. In 2011, the top-five accuracy of the winning model, based on classical approaches to computer vision, was only 74.3%. Then, in 2012, a team led by Alex Krizhevsky and advised by Geoffrey Hinton was able to achieve a top-five accuracy of 83.6% — a significant breakthrough. The competition has been dominated by deep convolutional neural networks every year since. By 2015, the winner reached an accuracy of 96.4%, and the classification task on ImageNet was considered to be a completely solved problem.

Since 2012, deep convolutional neural networks (convnets) have become the go-to algorithm for all computer vision tasks; more generally, they work on all perceptual tasks. At major computer vision conferences in 2015 and 2016, it was nearly impossi- ble to find presentations that didn’t involve convnets in some form. At the same time, deep learning has also found applications in many other types of problems, such as natural-language processing. It has completely replaced SVMs and decision trees in a wide range of applications.

### What's Different This Time?

The two key ideas of deep learning for computer vision—convolutional neural net- works and backpropagation—were already well understood in 1989. The Long Short- Term Memory (LSTM) algorithm, which is fundamental to deep learning for timeseries, was developed in 1997 and has barely changed since. So why did deep learning only take off after 2012? What changed in these two decades? In general, there are three driving forces:

* **Hardware**: GPU and increasing computational power from gaming industry. GPU adoption and adaptation to DL training with NVIDIA's CUDA, a programming interface for its line of GPUs. Google's dedicated TPU project.

* **Datasets and benchmarks**: ImageNet as an example: 1.4 million images that have been hand annotated with 1,000 image categories (1 category per image). But what makes ImageNet special isn’t just its large size, but also the yearly competition associated with it. As Kaggle has been demonstrating since 2010, public competitions are an excel- lent way to motivate researchers and engineers to push the envelope. Having common benchmarks that researchers compete to beat has greatly helped the recent rise of deep learning. This is what David Donoho argued to be the biggest driving success factor of ML compared to Statistics.

* **Algorithm**: Until the late 2000s, we were missing a reliable way to train very deep neural networks. The key issue was that of gradient propagation through deep stacks of layers. The feedback signal used to train neural network would fade away as a the number of layers increased. This changed around 2009–2010 with the advent of several simple but important algorithmic improvements that allowed for better gradient propagation:
	* Better _activation functions_ such as RELU
	* Better _weight-initialization schemes_ such as Xavier initialization or batch normalization
	* Better _optimization schemes_, such as RMSprops, ADAM, etc.

Because the field is guided by experimental findings rather than by theory, algorithmic advances only become possible when appropriate data and hardware are available to try new ideas (or scale up old ideas, as is often the case). Machine learning isn’t mathematics or physics, where major advances can be done with a pen and a piece of paper. It’s an **engineering science**.

## The Modern Machin Learning Landscape

A great way to get a sense of the current landscape of machine-learning algorithms and tools is to look at machine-learning competitions on Kaggle. Due to its highly competitive environment (some contests have thousands of entrants and million dollar prizes) and to the wide variety of machine-learning problems covered, Kaggle offers a realistic way to assess what works and what doesn’t. So, what kind of algorithm is reliably winning competitions? What tools do top entrants use?

In 2016 and 2017, Kaggle was dominated by two approaches: gradient boosting machines and deep learning. Specifically, gradient boosting is used for problems where structured data is available, whereas deep learning is used for perceptual problems such as image classification. Practitioners of the former almost always use the excellent XGBoost library, which offers support for the two most popular languages of data science: Python and R. Meanwhile, most of the Kaggle entrants using deep learn- ing use the Keras library, due to its ease of use, flexibility, and support of Python.

These are the two techniques you should be the most familiar with in order to be successful in applied machine learning today: gradient boosting machines, for shallow- learning problems; and deep learning, for perceptual problems. In technical terms, this means you’ll need to be familiar with XGBoost and Keras—the two libraries that currently dominate Kaggle competitions.

# Takeaways

* AI as a field has gone through a few hype cycles, we are in the third wave of this cycle, starting from 2012. This also happens to be the time that I left school, where the majority of my training is in shallow learning, rather than deep learning.
* At of the heart of this cycle is the development of Deep Learning, driven by hardware, dataset and benchmark, as well as algorithmic development
* Deep Learning differs from shallow learning is that representation of data are learned in successive layers, and jointly by taking information from previous and next layers
* Deep Learning at its current form and field is fundamentally an "engineering science", not a theoretical field where theories are developed using pen and pencils. They are developed from trying things out with compute and engineering.
* The current landscape in Kaggle competition is that XGBoost is the best algo for structure data, and Convnet are the best for perception tasks.

# Historical Fun Facts

* Geoffrey Hinton discovered backward propagation in the 1980s
* Yecunn developed LeNet which are used by USPS to recognized digits in the 1990s
* GPUs, originally developed for Gaming industry/application, became the hardware innovation for DL
* Hinton and his group start making breakthrough for ImageNet competition in 2012s
* Google dedicated to developed specialized chips, TPU, for DL training
* George Hinton introduced RMSprop in his Coursera course, and everyone cites the Coursera reference in their papers
* Inception Network was inspired by the Meme of Movie Inception