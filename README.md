# Physics-Informed Neural Networks

This repository presents a generalization of the *physics informed neural network* framework presented in [[1]](https://www.sciencedirect.com/science/article/pii/S0021999118307125) to solve partial differential equations.

## Work Summary
The implementation is done in **PyTorch** and incloudes the following features : 
* Train/evaluate pipeline to solve differential equations using the PINN framework.
* Optimization with either Adam or LBFGS.
* Batched data loading.
* Hybrid systems approach to solve problems involving multiple differential equations.

## Example 1 : Schrodinger's equation
 We solve the following problem for $h$ such that : 

 $ih_t + 0.5h_{xx} + |h|^2h = 0,\text{ for } x \in [-5, 5], t \in [0, \pi/2],\ [0]$

and :

 $h(0, x) = 2sech(x), [1]$

 $h(t, -5) = h(t, 5), [2]$

 $h_x(t, -5) = h_x(t, 5), [3]$ 

 For training, the model is fed $N_0 = 50$ points to sample $[1]$, $N_b = 50$ points to sample $[2]$ and $N_f = 20000$ points to make sure that [0] is respected.

 Results obtained can be summarized in the following figure : 
 
 ![Solution to Schrodinger's equation](figures/solution_test.gif)