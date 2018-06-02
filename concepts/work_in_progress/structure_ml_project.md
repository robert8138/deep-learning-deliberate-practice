# Structuring Machine Learning Project

_Notes from this section are adpated from Andrew Ng's DL specialization course 3 + Andrew's Machine Learning Yearning Book_, many of the notes here are copied verbatim, all rights belong to Andrew Ng.

## Table of Contents

* [Motivation](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#motivation)
* [Setting Up Development & Test Set](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#setting-up-development--test-set)
	* [Training, Validation, and Test Set](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#training-validation-and-test-set)
	* [Your Dev and Test Sets Should Come From the Same Distribution](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#your-dev-and-test-sets-should-come-from-the-same-distribution)
	* [Training Data Distribution Need Not To Be The Same as Val/Test Set Distribution](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#training-data-distribution-need-not-to-be-the-same-as-valtest-set-distribution)
	* [How Large Do Dev / Test Set Need to Be?](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#how-large-do-dev--test-set-need-to-be)

* [Evaluation Metrics](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#evaluation-metrics)
	* [Dev set / Metric Combo](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#dev-setmetric-combo)

* [Finding Where Model Is Making Mistakes](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#finding-where-model-is-making-mistakes)
	* [Error Analysis on Dev Set](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#error-analysis-on-dev-set)
	* [Cleaning Up Mislabeled Dev and Test Set Examples](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#cleaning-up-mislabeled-dev-and-test-set-examples)
	* [Break Dev Set to Eyeball Dev Set & Blackbox Dev Set](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#break-dev-set-to-eyeball-dev-set--blackbox-dev-set)

* [Bias and Variance](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#bias-and-variance)
	* [Comparing to Optimal Error Rate](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#comparing-to-optimal-error-rate)
	* [Addressing Bias and Variance](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#addressing-bias-and-variance)
		* [Addressing Bias Problems](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#addressing-bias-problems)
		* [Addressing Variance Problems](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#addressing-variance-problems)
	* [Yet Another Tool to Diagnoise Bias v.s. Variance: Learning Curve](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/work_in_progress/structure_ml_project.md#yet-another-tool-to-diagnoise-bias-vs-variance-learning-curve)



## Motivation

Often, when tackling a machine learning project, our model performance is not as good as we hoped for. In situations like this, there are many things that we can try (imagine we are building a cat classifier):

* Get more data: Collect more pictures of cats.
* Collect a more diverse training set. For example, pictures of cats in unusual positions; cats with unusual coloration; pictures shot with a variety of camera settings
* Train the algorithm longer, by running more gradient descent iterations
* Try a bigger neural network, with more layers/hidden units/parameters
* Try a smaller neural network
* Try adding regularization (such as L2 regularization)
* Change the neural network architecture (activation function, number of hidden units, etc.)
* ...

There are so many possibilities. Zayad Enam pointed out, in his [blog post](http://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html), why machine learning is 'hard' - His main argument is that debugging machine learning model is much harder and more time consuming. In his own words:

Wisdome from Andrew:

> The difficulty is that machine learning is a fundamentally hard debugging problem. Debugging for machine learning happens in two cases: 1) your algorithm doesn't work or 2) your algorithm doesn't work well enough. What is unique about machine learning is that it is ‘exponentially’ harder to figure out what is wrong when things don’t work as expected. Compounding this debugging difficulty, there is often a delay in debugging cycles between implementing a fix or upgrade and seeing the result. Very rarely does an algorithm work the first time and so this ends up being where the majority of time is spent in building algorithms.

Andrew's practical advice tries to provide guidance for making us **better ML problem debuggers**. This is perhaps one of the most important lessons that novice people can learn for structuring ML projects.


## Setting Up Development & Test Set

### Training, Validation, and Test Set

As we all already know, we typically split our 'data' into three parts:

* Training set​ — Which you run your learning algorithm on.
* Dev (development) set​ — Which you use to tune parameters, select features, and make other decisions regarding the learning algorithm. Sometimes also called the hold-out cross validation set​.
* Test set​ — which you use to evaluate the performance of the algorithm, but not to make any decisions regarding what learning algorithm or parameters to use.

In typical textbook example or tutorial, we would just randomize a .csv and split them into training, validation, and test set. However, thinking carefully about the distribution of these three sets is very important. In fact, sometimes the distribution may (or need) not to be the same.

### Your Dev and Test Sets Should Come From the Same Distribution

**Example**: You have your cat app image data segmented into four regions, based on your largest markets: (i) US, (ii) China, (iii) India, and (iv) Other. To come up with a dev set and a test set, say we put US and India in the dev set; China and Other in the test set. This is very problematic.

Once you define the dev and test sets, your team will be focused on improving dev set performance. Thus, the dev set should reflect the task you want to improve on the most: To do well on all four geographies, and not only two.

There is a chance that your team will build something that works well on the dev set, only to find that it does poorly on the test set. When dev and test set come from the same distribution, and dev set performance is clearly much better than test set, then you clearly have **overfit** to the dev set and/or dev set is not representative enough of the reality, the solution is to get more dev set data and retrain. However, When the distribution of dev and test sets are different, debugging becomes a lot harder, and several scenarios are possible:

* You have overfit to the dev set
* You haven't overfit to the dev set, but the test set is harder than the dev set. So your algorithm might be doing as well as could be expected, and no further significant improvement is possible.
* You haven't overfit to the dev set, and test set is not necessarily harder, just different. So what works well on the dev set just does not work well on the test set. In this case, a lot of your work to improve dev set performance might be wasted effort.

Working on machine learning applications is hard enough. Having mismatched dev and test sets introduces additional uncertainty about whether improving on the dev set distribution also improves test set performance. Having mismatched dev and test sets makes it harder to figure out what is and isn’t working, and thus makes it harder to prioritize what to work on. Avoid this!

### Training Data Distribution Need Not To Be The Same as Val/Test Set Distribution

The purpose of the dev and test sets are to direct your team toward the most important changes to make to the machine learning system​. So, you should do the following:

> Choose dev and test sets to reflect data you expect to get in the future
and want to do well on. 

ANdrew uses the analogy of Bulls-eye & Arrows. The location of the Bullseye is the distribution where our dev/test should reflect the reality. The performance metric that we are trying to optimize is our arrow that allows to see how close we are. If we have the wrong distribution for val/test set that doesn't reality, it's like we put the bulls-eye in the wrong position, and optimizing and shooting an arrow to that bulls-eye will not achieve our goal.

In other words, your test set should not simply be 30% of the available data, especially if you expect your future data (e.g. mobile phone images) to be different in nature from your training set (e.g. website images). This is a pretty important takeaways, especially transfer learning has become quite common place in the deep learning world.

Wisdome from Andrew:

> It requires judgment to decide how much to invest in developing great dev and test sets. But don’t assume your training distribution is the same as your test distribution. Try to pick test examples that reflect what you ultimately want to perform well on, rather than whatever data you happen to have for training.

### How Large Do Dev / Test Set Need to Be?

The dev set should be large enough to detect differences between algorithms that you are trying out. For example, if classifier A has an accuracy of 90.0% and classifier B has an accuracy of 90.1%, then a dev set of 100 examples would not be able to detect this 0.1% difference (the smallest difference would be at least 1 / 100 = 1%). Compared to other machine learning problems I’ve seen, a 100 example dev set is small. Dev sets with sizes from 1,000 to 10,000 examples are common. With 10,000 examples, you will have a good chance of detecting an improvement of 0.1%. In industry, it's not uncommon to try to improve the model performance by 0.01%.

How about the size of the test set? It should be large enough to give high confidence in the overall performance of your system. One popular heuristic had been to use 30% of your data for your test set. This works well when you have a modest number of examples—say 100 to 10,000 examples. But in the era of big data where we now have machine learning problems with sometimes more than a billion examples, the fraction of data allocated to dev/test sets has been shrinking, even as the absolute number of examples in the dev/test sets has been growing. There is no need to have excessively large dev/test sets beyond what is needed to evaluate the performance of your algorithms.

## Evaluation Metrics

During development, your team will try a lot of ideas about algorithm architecture, model parameters, choice of features, etc. Having a **single-number evaluation metric** for **optimization**​ such as accuracy allows you to sort all your models according to their performance on this metric, and quickly decide what is working best.

The important thing to consider here is whether your single evaluation metric has degenerate cases where it doesn't really demonstrate how good the model is. For example, for binary classification, if the labels are extremely imbalance, predicting everything to be the majority class would give us a highly accurate model, using that single evaluation metric. In this situation, we can be misguided.

If you have other important metrics that you would like to take into consideration, one way to take this into account is to make these secondary metrics as **satisficing metric** — your classifier just has to be “good enough” on this metric (e.g. latency can be a good satisficing metric).

Having a single-number evaluation metric speeds up your ability to make a decision when you are selecting among a large number of classifiers. It gives a clear preference ranking among all of them, and therefore a clear direction for progress.

### Dev set/Metric Combo

Andrew typically ask his teams to come up with an initial dev/test set and an initial metric in less than one week—rarely longer. It is better to come up with something imperfect and get going quickly, rather than overthink this. For mature applications, it might take longer to come up with these.

If you later realize that your initial dev/test set or metric missed the mark, by all means change them quickly. For example, if your dev set + metric ranks classifier A above classifier B, but your team thinks that classifier B is actually superior for your product, then this might be a sign that you need to change your dev/test sets or your evaluation metric.

There are a few reasons why, despite working very hard to optimize dev set using the optimization metric you have chosen, your model still doesn't work in reality:

* **You have overfit to the dev set**: The process of repeatedly evaluating ideas on the dev set causes your algorithm to gradually “overfit” to the dev set. When you are done developing, you will evaluate your system on the test set. If you find that your dev set performance is much better than your test set performance, it is a sign that you have overfit to the dev set. In this case, get a fresh dev set.

* **The actual distribution you need to do well on is different from the dev/test sets**: Suppose your initial dev/test set had mainly pictures of adult cats. You ship your cat app, and find that users are uploading a lot more kitten images than expected. So, the dev/test set distribution is not representative of the actual distribution you need to do well on. In this case, update your dev/test sets to be more representative.

* **The metric is measuring something other than what the project needs to optimize**: Suppose that for your cat application, your metric is classification accuracy. This metric currently ranks classifier A as superior to classifier B. But suppose you try out both algorithms, and find classifier A is allowing occasional pornographic images to slip through. Even though classifier A is more accurate, the bad impression left by the occasional
pornographic image means its performance is unacceptable. Here, the metric is failing to identify the fact that Algorithm B is in fact better than Algorithm A, because you cannot afford to show porn pictures. One way to combat this is to heavily penalize when porn pictures are surfaced.

> It is quite common to change dev/test sets or evaluation metrics during a project. Having an initial dev/test set and metric helps you iterate quickly. If you ever find that the dev/test sets or metric are no longer pointing your team in the right direction, it’s not a big deal! Just change them and make sure your team knows about the new direction.

## Finding Where Model Is Making Mistakes

### Error Analysis on Dev Set

**Error Analysis**​ refers to the process of examining dev set examples that your algorithm misclassified, so that you can understand the underlying causes of the errors. This can help you prioritize projects—as in this example—and inspire new directions. Here is the picture of how to carry that out:

In detail, here’s what you can do:
1. Gather a sample of 100 dev set examples that your system misclassified. i.e., examples that your system made an error on.
2. Look at these examples manually, and count what fraction of them belong to the issue class that you are trying to fix.

If you have a few ideas what's causing low performance, you can efficiently evaluate all of these ideas in parallel. Andrew usually create a spreadsheet and fill it out while looking through 100 misclassified dev set images. Andrew also jot down comments that might help me remember specific examples. To illustrate this process, let’s look at a spreadsheet you might produce with a small dev set of four examples:

![Error Analysis Table](pictures/error_analysis_table.png)

The most helpful error categories will be ones that you have an idea for improving. For example, the Instagram category will be most helpful to add if you have an idea to “undo” Instagram filters and recover the original image. But you don’t have to restrict yourself only to error categories you know how to improve; the goal of this process is to build your intuition about the most promising areas to focus on.

Error analysis does not produce a rigid mathematical formula that tells you what the highest priority task should be. You also have to take into account how much progress you expect to make on different categories and the amount of work needed to tackle each one.

Wisdome from Andrew:

> Error analysis can often help you figure out how promising different directions are. I’ve seen many engineers reluctant to carry out error analysis. It often feels more exciting to just jump in and implement some idea, rather than question if the idea is worth the time investment. This is a common mistake: It might result in your team spending a month only to realize afterward that it resulted in little benefit.

### Cleaning Up Mislabeled Dev and Test Set Examples

During error analysis, you might notice that some examples in your dev set are mislabeled. When I say “mislabeled” here, I mean that the pictures were already mislabeled by a human labeler even before the algorithm encountered it. I.e., the class label in an example (x,y) has an incorrect value for y. For example, perhaps some pictures that are not cats are mislabeled as containing a cat, and vice versa. If you suspect the fraction of mislabeled images is significant, add a category to keep track of the fraction of examples mislabeled.

This is actually a very common scenario when you have scarce labels, or that labels are generated by some heuristics (e.g. For earlier version of the room classification model, we use caption data to generate label before we move on to using third party labelers).

![Error Analysis Table 2](pictures/error_analysis_table_2.png)

Should you correct the labels in your dev set? Remember that the goal of the dev set is to help you quickly evaluate algorithms so that you can tell if Algorithm A or B is better. If the fraction of the dev set that is mislabeled impedes your ability to make these judgments, then it is worth spending time to fix the mislabeled dev set labels.

Whatever process you apply to fixing dev set labels, remember to apply it to the test set labels too so that your dev and test sets continue to be drawn from the same distribution. Fixing your dev and test sets together would prevent the problem we discussed earlier, where your team optimizes for dev set performance only to realize later that they are being judged on a different criterion based on a different test set.

Wisdome from Andrew:

> If you decide to improve the label quality, consider double-checking both the labels of examples that your system misclassified as well as labels of examples it correctly classified. It is possible that both the original label and your learning algorithm were wrong on an example. If you fix only the labels of examples that your system had misclassified, you might introduce bias into your evaluation. If you have 1,000 dev set examples, and if your classifier has 98.0% accuracy, it is easier to examine the 20 examples it misclassified than to examine all 980 examples classified correctly. 

### Break Dev Set to Eyeball Dev Set & Blackbox Dev Set

Suppose you have a large dev set of 5,000 examples in which you have a 20% error rate. Thus, your algorithm is misclassifying 1,000 dev images. It takes a long time to manually examine 1,000 images, so we might decide not to use all of them in the error analysis.

In this case, Andrew suggests to explicitly split the dev set into two subsets, one of which you look at (called **eyeball-dev set**), and one of which you don’t (called **blackbox-dev set**). When using eyeball-dev set for error analysis, you are more likely to overfit the portion that you are manually looking at. By setting aside blackbox-dev set, we can understand the extent in which we are overfitting on the manually analyzed examples.

How large should the eyeball-dev set be? General rule of thumb is to have 100 random misclassified samples, anything smaller is typically too small, and anything > 1000 might be too time consuming. A good back of the envelope calculation is to take 100 / misclassification rate to be the size of the eyeball-dev set. For example, if the misclassification rate is 5%, then we need 2000 examples in eyeball-dev set. The more accurate the model, the more eyeball-dev set we will need.

Wisdom from Andrew:

> Between the Eyeball and Blackbox dev sets, I consider the Eyeball dev set more important (assuming that you are working on a problem that humans can solve well and that examining the examples helps you gain insight). If you only have an Eyeball dev set, you can perform error analyses, model selection and hyperparameter tuning all on that set. The downside of having only an Eyeball dev set is that the risk of overfitting the dev set is greater.

## Bias and Variance

There are two major sources of error in machine learning: bias and variance. Understanding them will help you decide whether adding data, as well as other tactics to improve performance, are a good use of time. 

The field of statistics has more formal definitions of bias and variance. Roughly, the bias is the error rate of your algorithm on your training set when you have a very large training set (i.e. how complex your functional form will fit on the training data). The variance is how much worse you do on the test set compared to the training set (for me, I like to think of it as how robust the model is to variance introduce by sampling / and being able to see the whole reality distribution, Caltech's ML course really explains well). See chapter 21 for scenario where there is bias and/or variance problems.

Some changes to a learning algorithm can address the first component of error-bias​ and improve its performance on the training set. Some changes address the second component—variance and help it generalize better from the training set to the dev/test sets. To select the most promising changes, it is incredibly useful to understand which of these two components of error is more pressing to address.

### Comparing to Optimal Error Rate

To debug whether our learning task is sufferring from bias and/or variance problem. It's very useful to have **Optimal Error Rate (Bayes Error Rate)** as a baseline. Andrew called optimal error rate **Unavoidable Bias** - these are just errors that we are going to make, regardless how good the model is. The classic example is speech recognition, some video clips would have so much background noise that there is no way to tell what is the conversation.

To really fit optimal error rate into the bias-variance discussion, Andrew propose the following:

> Error = Bias + Variance = Optimal error rate (“unavoidable bias”) + Avoidable bias + Variance

The “avoidable bias” reflects how much worse your algorithm performs on the training set than the “optimal classifier.” The concept of variance remains the same as before. In theory, we can always reduce variance to nearly zero by training on a massive training set. Thus, all variance is “avoidable” with a sufficiently large dataset, so there is no such thing as “unavoidable variance.”

How do we know what the optimal error rate is? For tasks that humans are reasonably good at, such as recognizing pictures or transcribing audio clips, you can ask a human to provide labels then measure the accuracy of the human labels relative to your training set. This would give an estimate of the optimal error rate. If you are working on a problem that even humans have a hard time solving (e.g., predicting what movie to recommend, or what ad to show to a user) it can be hard to estimate the optimal error rate.

### Addressing Bias and Variance

Here is the simplest formula for addressing bias and variance issues:

* If you have high avoidable bias, increase the size of your model (for example, increase the size of your neural network by adding layers/neurons).
* If you have high variance, add data to your training set.

If you are able to increase the neural network size and increase training data without limit, it is possible to do very well on many learning problems. This is one of the reasons why DL works so well - many techniques allowed us to trained very deep and wide networks, and there's enough data (at least from companies like Google) for us to dramatically reduce bias and variance of learning tasks. In practice, increasing the size of your model will eventually cause you to run into computational problems because training very large models is slow. You might also exhaust your ability to acquire more training data. 

Wisdom from Andrew:

> Different model architectures—for example, different neural network architectures will have different amounts of bias/variance for your problem. A lot of recent deep learning research has developed many innovative model architectures. So if you are using neural networks, the academic literature can be a great source of inspiration. There are also many great open-source implementations on github. But the results of trying new architectures are
less predictable than the simple formula of increasing the model size and adding data. If you increase model size, you will run into the risk of overfitting. In such cases, add regularization accordingly.

#### Addressing Bias Problems

* **Increase the model size ​(such as number of neurons/layers)**: This technique reduces bias, since it should allow you to fit the training set better. If you find that this increases variance, then use regularization, which will usually eliminate the increase in variance.

* **Modify input features based on insights from error analysis**​: Say your error analysis inspires you to create additional features that help the algorithm eliminate a particular category of errors. These new features could help with both bias and variance. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance.

* **Reduce or eliminate regularization**: This will reduce bias, but increase variance.

* **Performing Error Analysis on Training Set**: Just like how we would do this analysis on dev set, analyzing wrong predictions on the training set can also help us understand why the model is not performing well.

#### Addressing Variance Problems

* **Add more training data**: This is the simplest and most reliable way to address variance, so long as you have access to significantly more data and enough computational power to process the data. More data -> more opportunities to see more patterns.

* **Add regularization​ (L2 regularization, L1 regularization, dropout)**: This technique reduces variance but increases bias.

* **Add early stopping​ (i.e., stop gradient descent early, based on dev set error)**: This technique reduces variance but increases bias. Early stopping behaves a lot like regularization methods, and some authors call it a regularization technique. Andrew doesn't like this because it violates the principle of orthogonalization.

* **Feature selection to decrease number/type of input features**:​ This technique might help with variance problems, but it might also increase bias. Reducing the number of features slightly (say going from 1,000 features to 900) is unlikely to have a huge effect on bias. Reducing it significantly (say going from 1,000 features to 100 — a 10x reduction) is more likely to have a significant effect, so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful.

* **Decrease the model size ​(such as number of neurons/layers)**: Use with caution. This technique could decrease variance, while possibly increasing bias. However, Andrew doesn't recommend this technique for addressing variance. Adding regularization usually gives better classification performance. The advantage of reducing the model size is reducing your computational cost and thus speeding up how quickly you can train models. If speeding up model training is useful, then by all means consider decreasing the model size. But if your goal is to reduce variance, and you are not concerned about the computational cost, consider adding regularization instead.

### Yet Another Tool to Diagnoise Bias v.s. Variance: Learning Curve

We’ve seen some ways to estimate how much error can be attributed to avoidable bias vs.variance. We did so by estimating the optimal error rate and computing the algorithm’s training set and dev set errors. Let’s discuss a technique that is even more informative: plotting a learning curve.

Previously, we were measuring training and dev set error only at the rightmost point of this plot, which corresponds to using all the available training data. Plotting the full learning curve gives us a more comprehensive picture of the algorithms’ performance on different training set sizes.

![Learning Curve Scenarios](pictures/learning_curve_scenarios.png)

In the left plot, we see that training error is way above the desired performance (this might be optimal error rate, human level performance, or something that the product team believe to be acceptable). When training size increases, we know that training error would typicall increase. Given that we know that dev set error is usually above training error, increasing data size will not help us to reduce dev set error. The big gap between optimal error rate & training error suggests that we have a bias problem, the small gap between training error and dev set error suggests the variance problem is small. 

In the right plot, training error is comparable to the desired error rate, so there is no bias problem. However, dev set error is much bigger than training error, this suggests that we might have overfit to the training set. In such case, we have low bias, high variance problem. Adding more data points to training would help.

In reality, plotting learning curve and understanding the pattern can be tricky. For example:

* **Small sample size creates noise in measurement**: When training on just 10 randomly chosen examples, you might be unlucky and have a particularly “bad” training set, such as one with many ambiguous/mislabeled examples. Or, you might get lucky and get a particularly “good” training set. Having a small training set means that the dev and training errors may randomly fluctuate.

* **Skew Class / Too Many Classes**: If your machine learning application is heavily skewed toward one class (such as a cat classification task where the fraction of negative examples is much larger than positive examples), or if it has a huge number of classes (such as recognizing 100 different animal species), then the chance of selecting an especially “unrepresentative” or bad training set is also larger. For example, if 80% of your examples are negative examples (y=0), and only 20% are positive examples (y=1), then there is a chance that a training set of 10 examples contains only negative examples, thus making it very difficult for the algorithm to learn something meaningful.

To combat these, we can either sample with replace and plot the average training & dev error to smooth out noise, or we can fix the distribution of the classes in the dataset. 

Wisdom from Andrew:

> I would not bother with either of these techniques (sampling with replacement or fix distribution of classes) unless you have already tried plotting learning curves and concluded that the curves are too noisy to see the underlying trends. If your training set is large—say over 10,000 examples—and your class distribution is not very skewed, you probably won’t need these techniques. When you have a lot of data, you don't have to space the training sample size in linear scale. A log scale can make trends clearer.























## Misc


[**How to think about problem formulation**](http://cs230.stanford.edu/files/Week2_slides.pdf)
	- Input: What is the the input of the model, number of inputs, size of input
	- Output: What is the output of the model, number of outputs
	- Data: What data do you have?
	- Architecture: Shallow network, Deep network (standard or customized)
	- Loss: How would you optimize the loss of your learning problems
	- Training: What is the training process learning? parameters / input images?

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