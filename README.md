# ManiML

My attempt at using the manim engine to animate the numerical optimization process for a bunch of machine learning problems. For now there is linear regression, and linear classification using logistic regression. Nothing too extraordinary but it shows how the model and the loss behave as the algorithm makes its way through the data.

## 1. Linear Regression

The idea is simple, you've got a bunch of data and you need to find a line or a plane or a hyperplane, depending on how many dimensions you have that passes closest to the data points ie minimizes the average distance from the hyperplane to the points.

### 1.1 Batch gradient descent

You've got a cost function and you calculate its gradient w.r.t. model parameters, then you repeatedly step ever so slightly along the gradient till you converge at the minimum like a falcon upon its prey.

https://user-images.githubusercontent.com/63511027/212228949-48d70c6b-5e99-4fc1-97de-72e7534bfb3d.mp4

### 1.2 Stochastic gradient descent

Same thing as batch gradient descent except instead of calculating the cost function over the entire dataset you randomly pick one instance and calculate it over that single one. 

Here it can be noted that the loss doesn't decrease so smoothly like in BGD, also even after having found a good enough solution it kind of 'rocks' about the optimum a bit.

https://user-images.githubusercontent.com/63511027/212229000-df60f393-7089-43a0-8d49-a55f03793d94.mp4

## 2. Logistic Regression (Linear)

A bunch of data labelled into two classes (0, 1), in some space, you find a line(hyperplane) that separates them into their classes

### 2.1 Batch gradient descent

Same as the one in linear regression but uses a different loss function called log loss.

https://user-images.githubusercontent.com/63511027/212230301-a284170c-dfb8-42a2-ac1f-04a3610a80de.mp4


### 2.2 Stochastic gradient descent

Same as the one in linear regression but uses log loss function.

https://user-images.githubusercontent.com/63511027/212230726-c951d070-13aa-487b-889d-e89712dbb3b3.mp4
