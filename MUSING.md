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

# Embeddings

* Started off with NLP, where people are motivated to move from one-hot-encoding (sparse, inefficient) of words to embeddings (dense, more efficient) representation.

* There are different algorithms to learn embeddings,the most famous one being [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
	* See [CS 224N notes](http://web.stanford.edu/class/cs224n/lectures/lecture2.pdf) for detailed setup for the problem: objective function, cost function, loss ... etc
	* The idea is to trying to maximize the likelihood that a `context/outside` word appeared near the `center` word -> `max P(outside context word | center word)`
	* Concept of negative sampling, where we `max P(relevant outside/context w | center w)` and `min P(irrelevant outside/context w | center w)` => `max P(relevant outside/context w | center w) - \sum_j ^{k} P(irrelevant outside/context w | center w)`, the latter part is the negative sampling part.

* Interestingly, the word2vec approach of learning word embeddings can be extended to learning embeddings of other domains. In Mihajlo's word: More recently, the concept of embeddings has been extended beyond word representations to other applications outside of NLP domain. Researchers from the Web Search, E-commerce and Marketplace domains have realized that just like one can train word embeddings by treating a sequence of words in a sentence as context, the same can be done for training embeddings of user actions by treating sequence of user actions as context. Examples include learning representations of items that were clicked or purchased or queries and ads that were clicked. These embeddings have subsequently been leveraged for a variety of **recommendations** on the Web. <- This really opens a new door for me to think about building recommendation system. See our very own Airbnb blog post on [listing embeddings](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e). 

* Rachel Thomas actually wrote a post called ["category embedding"](http://www.fast.ai/2018/04/29/categorical-embeddings/) that talks about similar concepts.