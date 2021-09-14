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

 ##  State of the art

 The idea of using neural networks (NN) to solve Ordinary Differential Equations (ODEs) and Partial Differential Equations (PDEs) has been widely explored for a long time. The first publication trying to do it was written by I.E. Lagaris et al. in 1997 [[1]](#1). Neural networks are used with a custom loss function which ensures that the ODE or PDE is satisfied by the NN. An aditional term is added to ensure boundary and initial conditions.

The term Physically-Informed Neural Network (PINN) comes from the fact that the loss function contains a specific term for the physical constraints of the system studied [[2]](#2). With this term, symetries or asymetries can be learned by the NN. The loss is computed using the values and approximate derivatives of the predictions, ensuring the correctness of the solution with respect to the ODE or PDE.

This framework has recently been extended [[3]](#3) to include new features for specific problems. It can now deals with spatial domain decomposition for non-homogeneous environments and allows parallelization and localized representation over the different domains.

PINNs can be used to solve ODEs and PDEs, but we needed to design an algorithm to apply them to hybrid systems. A classical method for solving hybrid systems is described by Fu Zhang et al. [[4]](#4). It consists in solving the equations until a guard condition is met, then switching states and computing the behavior of the system with the new conditions. We can use PINNs instead of traditional numerical method to solve the equations until a guard condition is met, then go on in the new state.

We expect to find the same limitations and issues discussed by Krishnapriyan et al. [[5]](#5). PINNs can create problems when some parameters of the equations are too large and the errors can be locally instable around the training points, which we need to take care of during our training. Also, in [[4]](#4), a specific behavior called Zeno (characterized by an infinite number of transitions in a finite time), seems to be of concern, and may not be detected using our method.

## References

<a id="1">[1]</a> 
I. E. Lagaris, A. Likas, D. I. Fotiadis (1997).
Artificial Neural Networks for Solving Ordinary and Partial Differential Equations.

<a id="2">[2]</a>
M. Raissi, P. Perdikaris, G.E. Karniadakis (2019).
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
Journal of Computational Physics,
Volume 378,
2019,
Pages 686-707.

<a id="3">[3]</a> 
Jagtap, Ameya & Karniadakis, George. (2020).
Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.
Communications in Computational Physics. 28. 2002-2041. 10.4208/cicp.OA-2020-0164.

<a id="4">[4]</a> 
Fu Zhang, Murali Yeddanapudi, Pieter J. Mosterman (2008).
Zero-Crossing Location and Detection Algorithms For Hybrid System Simulation,
IFAC Proceedings Volumes,
Volume 41, Issue 2,
2008

<a id="5">[5]</a> 
Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney (2021).
Characterizing possible failure modes in physics-informed neural networks.

 