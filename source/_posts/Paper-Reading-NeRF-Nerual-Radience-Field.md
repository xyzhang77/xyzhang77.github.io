---
layout: pages
title: 'Paper Reading: NeRF(Nerual Radience Field)'
date: 2022-04-03 13:03:15
tags: '3D Representation implicitly'
categories: 
    - ['Paper Reading']
---

![NeRF](https://s2.loli.net/2022/05/05/CEr1ZX8dDmIak4P.png)  


## Introduction
NeRF is to solve the problem of view sythesis by directly optimizing parameters of a continuous 5D function,
and it uses neural network to represent scenes and store the information of a real model. The neural network has a 
5D input, the spatial location $(x, y, z)$ and viewing direction $(\theta, \phi)$, and outputs the volume density
and the view-dependent emitted radiance at that spatial location. The volume density is only related with the spatial
location, while radiance is also related with the viewing direction. It is because we can see different color from 
different view, like mirrors. 

<!-- more -->
## Neural Radiance Field Scene Representation

The output can be seen as a vector in a 3D point, and there's a vector field called neural radiance field.

The input is 3D location ${\bf{x}}=(x,y,z)$ and 2D view
direction $(\theta, \phi)$ which also can be represented as 3D Cartesian unit vector ${\bf{d}}$. The structure of 
the network is extremely simple and is an 8-layer MLP with a skip connection at 4th layer using ReLU activations and 256 channels per layer. The output is 
$\sigma$ which is the density, and  the 256-dimension 
feature. Then the feature vector is concatenated with 
view direction, and then fed into another MLP which 
outputs the view-dependent RGB color.

## Volume Rendering

What we want to do is synthesize novol view only by camera intrinsics and extrinsics. But the input is not
the parameters. 

Actually, the color of each pixel can be seen as how much 
radiance arrive at the image plane along the ray that emits from the camera and goes through the specific pixel.

So we can formulate the color of camera as 

$$
C({\bf{r}}) = \int_{t_n}^{t_f}  T(t)\sigma({\bf r}(t)){\bf c}({\bf r}(t),{\bf d}) \mathrm{d}t, \text{where}\ T(t) = \exp(-\int_{t_n}^t\sigma({\bf r}(s))\  \mathrm{d}s)
$$

The function $T(t)$ denotes the accumulated transmittance along the ray from $t_n$ to $t$, the probility taht the ray travels from
$t_n$ to $t$ without hitting any other particle. If there's $t$ such that $\sigma$ is large, $T$ will drop drastically, which 
is as following.

![The red line is T, the blue is sigma](https://s2.loli.net/2022/05/05/IOzwu7M4gpK2PGH.png)  

We need to estimate the above fucntion using a numerical method. And we sample the rays discretize the integral, and use left Riemann sum. So the integral can be represented as 

$$
\hat{C}({\bf{r}}) = \sum_{i=1}^N T_i(1-exp(-\sigma_i\delta_i)){\bf c}_i,\ \text{where}\ T_i = \exp(-\sum_{j=1}^{i-1} \sigma_j\delta_j)
$$

## Two Tricks in NeRF

### Positional Encoding

The input is a 5D coordiante, and it is too diffical to use this 5D data direcly estimate the geometry information especially 
the high-frequency information. They use positional encoding to map the original data to a higher dimension to enable better
fitting of data that contains high frequency variation

$$
\gamma(p) = (sin(2^0\pi p), cos(2^0\pi p), \cdots, sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))
$$

$$
F_{\Theta} = F_{\Theta}'\circ\gamma
$$

We also see positional encoding in Transformer, a model for NLP. Although they have the same name and function, 
they serve different purposes. In Transformer, it is to add positional information, because attention doesn't take position into consider, and here, it is to map the
data into higher dimension to easily approximate a higher frequency function.
### Hierarchical Sampling

In volume rendering, we approximate the integral by sampling points along the ray. The more sample points, the more precise it is. But it will also lead to higher overhead.
So the samping strategy is important.

Hierarchical sampling is a sampling strategy from coarse to fine. Uniformly sampling will treat each 
interval equally, however, we need sample more points in the interval that the volume is occupied by the object. 

They use two network, "coarse" and "fine". First, we use uniform sampling and feed in coarse network,
$$
\hat{C_c}({\bf r}) = \sum_{i=1}^{N_c}w_ic_i,\ w_i=T_i(1-\exp(-\sigma_i \delta_i))
$$ 

then normalize the weight. After that, use inverse sample to  get another sample sets.

## Loss Function

The input is 5D coordinate and output is color, and actually we have already know the color of the integral. So the loss is the distence between estimated color and expected color.

Due to the architecture of dual network, we have to add 
the color from the coarse network to train easiler.

So the loss function is 

$$
\mathcal{L}=\sum_{\mathbf{r} \in \mathcal{R}}\left[\left\|\hat{C}_{c}(\mathbf{r})-C(\mathbf{r})\right\|_{2}^{2}+\left\|\hat{C}_{f}(\mathbf{r})-C(\mathbf{r})\right\|_{2}^{2}\right]
$$