# Tropical time series - Accompanying code
Accompanying code for [Tropical time series, iterated-sums signatures and quasisymmetric functions](https://arxiv.org/abs/2009.08443).

## Experiments

The code generates a white noise signal of variable lengths and adds two different fixed patterns at random timestamps.
This generates two classes of signals, which are then classified using a Neural Network architecture consisting of a 
"tropical-sums layer" built on top of a Multilayer Perceptron.
The results of this classification are then compared with a plain Dense Network of variable architecture.

### Findings

We find that for this simple task, the Dense architecture overfits the training data producing a 100% training accuracy,
with a test accuracy of around 60% depending on the problem. In contrast, the Tropical-sums Network consistently produces
test accuracy close to 85% across all examples.
