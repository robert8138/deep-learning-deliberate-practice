# Backward Propogation

Once you understand the basic architecture of Neural Network, performing a forward pass from `x` -> `z` -> `a` -> `y` is relatively straightforward. This forward path merely describe, when we have a **trained model**, how the inference will be done. Backward propogation, on the other hand, describes exactly how it updates the parameters of the model `w` and `b` in a backward fashion.

The math for backprop can seem confusing, and that's because it involves a lot of matrix calculus and careful tracking of indices. I present (what I think is) the best approach to learn the math behind backprop, starting from a refresher on Matrix Calculus, then move on to proving it.

## Table of Contents

### Matrix Calculus & Proofs

* [Matrix Calculus for Deep Learning](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html)
* [Michael Nielsen's Detailed Explanation on Backprop](http://neuralnetworksanddeeplearning.com/chap2.html)
* [Justin Johnson's notes on Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/2017/handouts/derivatives.pdf)
* My own derivation of 4 fundamental equations for backprop is written [here](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/pictures/backprop_math_by_hand.png)

### Intuition on Backprop & Why Backprop Makes Training Hard

* [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)
* [Yes You Should Understand Backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
* [Why are deep neural networks hard to train?](http://neuralnetworksanddeeplearning.com/chap5.html)

## Matrix Calculus for Deep Learning

This document was written by Jeremey Howard, and it's a very detailed (but nice) treatment of Matrix Calculus. 

* First, Matrix Calculus arise when you need to take derivative of vector value function with vector value input - the derivative of f w.r.t input in this case is a Jacobian matrix. Depending on whether f is vector-valued, or if the input is vector-valued, this would change the derivative to a scalar, row vector, column vector, or a Jacobian matrix.

* Second, with the basic mechanics of Matrix Calculus mastered, he then discussed how chain rule works. The concept of total derivative comes up, which manifest itself a lot in matrix multiplication of Jacobian matrices based on the chain rule. This is important concept to master in order to correctly derive backprop in neural network

* Finally, he went through a simplified (one layer example) of deriving the `w` and `b` update in backprop

## Justin Johnson's Notes on Tensor Derivative

There are two things great about Justin's notes:

* He introduced Calculus as rate of change, and show how `x -> x + dx` would lead to `y -> y + dy/dx * dx`, which explains how to reason about the dimension of `dy/dx`, very helpful!
* He introduced generalized version of the Tensor Calculus, where the input and output can both be tensors. Derivatives and Chain rules still apply, but are more complex.

## Michael Nielsen's Detailed Explanation on Backprop

Unlike Colah's post, this long posts goes into detail explaining the math, notation, derivation of how backprop works in the context of Neural Networks. It's a really good read. To summarize, it:

* First introduced the notation (and explain why they are set up that way)

* Lay out the four fundamental equations for backprop:
	* An equation for the error in the output layer: `δL`
	* An equation for the error `δl` in terms of the error in the next layer, `δl+1`
	* An equation for the rate of change of the cost with respect to any bias in the network (`dCost/db`)
	* An equation for the rate of change of the cost with respect to any weight in the network (`dCost/dw`)

* It also goes into proving 2 of the 4 fundamental equations (I derived this myself, see [here](https://github.com/robert8138/deep-learning-deliberate-practice/blob/master/concepts/pictures/backprop_math_by_hand.png))
* It then lays out the backprop algorithm with SGD with mini-batch
* It shows Python code snippets for backprop
* It explains why it's much faster than forward propogation
* It then talks about a bit of the history of how backprop was discovered (Answer: a lot of hardwork in using forward pass reasoning and simplifying the math)

## Calculus on Computational Graphs

This post from Colah explains how, fundamentally, backpropogation is a technique for calculating derivatives quickly. And it’s an essential trick to have in your bag, not only in deep learning, but in a wide variety of numerical computing situations. It doesn't go into details in explaining the derivation of gradient for DL, but it explains why backprop is a much more efficient way to compute derivatives than forward propogation.

The key idea is this: when thinking about `dCost / dw`, where the lineage from `w` to `Cost` involves a lot of forward path, so the total derivative involves a lot of sum, the backward prop leverage the fact that `dCost/dw` can be thought of "backward" as `dCost/d local_variable * d local_variable / dw`. The beauty of the backward pass is that if we work backward, when we go from layer `l+1` to `l`, we will compute all the later graidents already, this makes `dCost/d local_variable * d local_variable / dw` a lot easier, and you only have to worry about local gradient, the upstream gradient `dCost/d local_variable` is taken care for you.

## Yes You Should Understand Backprop

A Medium post from Andre Karpathy explaining why every practitioners should learn about how backprop works. Essentially, his arguments is that even when doing DL in practice, if the activations are saturated, the gradient can become very close to 0 (vanishing gradient problem), causing the weight updates to be super slow, and in terms make the cost function to go down very slowly.
