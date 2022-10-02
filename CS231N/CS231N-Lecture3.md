# Lecture 3: Loss Functions and Optimization

[Lecture 3 on Youtube](https://youtu.be/h7iBpEHGVNc)

## Loss Functions 

### Multiclass Support Vector Machine (SVM) loss

#### Hinge loss
$$
L_i = \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)
$$

```python
def L_i_vectorized(x, y, W):
  scores = W.dot(x)
  margins = np.maximum(0, scores - scores[y] + 1)
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i
```

### Loss functions with Regularization term

$$
L = \frac{1}{N} \sum_{i=1}^N \sum_{j \ne y_i} \max(0, f(x_i ; W)_{y_j} + 1) + \lambda R(W)
$$

### Regularization

Necessary so that your models doesn't fit your training data too well.

- L2 regularization $R(W) = \sum_k \sum_l W^2_{k,l}$
- L1 regularization $R(W) = \sum_k \sum_l |W_{k,l}|$
- Elastic net (L1 + L2)
- Max norm regularization
- Dropout
- Fancier: Batch normalization, stochastic depth

### Softmax Classifier (Multinomial Logistic Regression)

#### Maximum Likelihood estimate 
$$
L_i = -\log (\frac{e^{s_{y_i}}}{\sum_j e^{s_j}})
$$

![Softmax Classifier Example](../Lecture3-Softmax.png)

### Recap

![Softmax Classifier Example](../Lecture3-Loss-Function-Recap.png)

## Optimization 

### Follow the slope
In 1-dimension, the derivative of a function:

Numerical Gradient:
$$
\frac{df(x)}{dx} = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}
$$

In multiple dimensions, the gradient is the vector of partial derivatives along each dimensions.

In practice we use the analytic gradient.

### Gradient Descent

```python
# Vanilla Gradient Descent
while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
```

`step_size` is a hyperparameter and determines the learning rate.


### Stochastic Gradient Descent (SGD)

Full sum is expensive when your dataset is large.

```python
# Vanilla Minibatch Gradient Descent

while True:
  # random sample of 265 examples
  data_batch = sample_training_data(data, 256) 
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```