---
layout: pages
title: Detail in reproducing VolSDF
date: 2022-04-19 11:38:39
tags:
---

This post mainly records the reproducing detail and helps me 
reproduce VolSDF more efficiently.

<!-- more -->
## Volume Rendering

### Signed Distence Function
$$
\mathbf{1}_{\Omega}(\boldsymbol{x})=\left\{\begin{array}{ll}
1 & \text { if } \boldsymbol{x} \in \Omega \\
0 & \text { if } \boldsymbol{x} \notin \Omega
\end{array}, \quad \text { and } d_{\Omega}(\boldsymbol{x})=(-1)^{\mathbf{1}_{\Omega}(\boldsymbol{x})} \min _{\boldsymbol{y} \in \mathcal{M}}\|\boldsymbol{x}-\boldsymbol{y}\|\right.
$$

In VolSDF, $d_{\Omega}(\boldsymbol{x})$ is estimated by network directly. 

> To satisfy the assumption that all rays, including rays that do not intersect any surface, are eventually occluded (i.e., O($\infty$) = 1), we model our SDF as: 
$$
d_\Omega(\boldsymbol x) = \min \{d(\boldsymbol x), r - ||x||_2\}
$$

### Density $\sigma$

$$
\sigma(\boldsymbol{x})=\alpha \Psi_{\beta}\left(-d_{\Omega}(\boldsymbol{x})\right)
$$
 
And, 

$$
\begin{aligned}
    \Psi_{\beta}(s) &= \begin{cases}\frac{1}{2} \exp \left(\frac{s}{\beta}\right) & \text { if } s \leq 0 \\ 1-\frac{1}{2} \exp \left(-\frac{s}{\beta}\right) & \text { if } s>0\end{cases} \\
    &= -\text{sign}(s)\cdot \frac{1}{2}\exp(-\frac{|s|}{\beta}) + \frac{1+\text{sign}(s)}{2} 
\end{aligned}
$$

where $\alpha, \beta > 0$ are learnable parameters, in experiments, $\alpha = \frac{1}{\beta}$ 

### Rendering Procedure

The rendering function is the same as the one in NeRF,

$$
I(\boldsymbol{c}, \boldsymbol{v})=\int_{0}^{\infty} L(\boldsymbol{x}(t), \boldsymbol{n}(t), \boldsymbol{v}) \tau(t) d t
$$

And 
$$
\tau(t)=\frac{d O}{d t}(t)=\sigma(\boldsymbol{x}(t)) T(t)
$$

$$
T(t)=\exp \left(-\int_{0}^{t} \sigma(\boldsymbol{x}(s)) d s\right)
$$
$\sigma$ is the density, $T$ is the probility that the ray travels from tn to t without hitting
any other particle.

The integral is approximated usign a numerical quadrature, namely the rectangle rule:
$$
I(\boldsymbol{c}, \boldsymbol{v}) \approx \hat{I}_{\mathcal{S}}(\boldsymbol{c}, \boldsymbol{v})=\sum_{i=1}^{m-1} \hat{\tau}_{i} L_{i}
$$

$$
\begin{aligned}
I(\boldsymbol{c}, \boldsymbol{v}) &=\int_{0}^{\infty} L(\boldsymbol{x}(t), \boldsymbol{n}(t), \boldsymbol{v}) \tau(t) d t \\
&=\int_{0}^{M} L(\boldsymbol{x}(t), \boldsymbol{n}(t), \boldsymbol{v}) \tau(t) d t+\int_{M}^{\infty} L(\boldsymbol{x}(t), \boldsymbol{n}(t), \boldsymbol{v}) \tau(t) d t \\
& \approx \sum_{i=1}^{m-1} \delta_{i} \tau\left(s_{i}\right) L_{i}
\end{aligned}
$$
$T$ is also a integral, so we use the same method to estimate it.
$$
T\left(s_{i}\right) \approx \hat{T}\left(s_{i}\right)=\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)
$$

$$
p_i = \exp(-\sigma_i\delta_i)
$$
Then , $\hat{T}(s_i)=\Pi_{j=1}^{i-1}p_j$, estimate $\sigma_i\delta_i$
$$
\sigma_i\delta_i \approx (1-\exp(-\sigma_i\delta_i)) = 1-p_i
$$
when $\sigma_i\delta_i$ is very small, the loss can be omitted.

<center>
<img src="https://blog-image-zxy.oss-cn-hangzhou.aliyuncs.com/
sigma_approx_f5d5850c.png" width="60%">
<br>
<div style="color:orange;solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;"><b>Blue line for δσ, red line for 1 - p</b></div>
</center>

So the $\tau$ can be written as following:
$$
\hat{\tau}_{i}=\left(1-p_{i}\right) \prod_{j=1}^{i-1} p_{j} \quad \text { for } 1 \leq i \leq m-1, \quad \text { and } \hat{\tau}_{m}=\prod_{j=1}^{m-1} p_{j}
$$
And 
$$
I(\boldsymbol{c}, \boldsymbol{v}) \approx \hat{I}(\boldsymbol{c}, \boldsymbol{v})=\sum_{i=1}^{m-1}\left[\left(1-p_{i}\right) \prod_{j=1}^{i-1} p_{j}\right] L_{i}
=\sum_{i=1}^{m-1}\left[\left(1-\exp(-\sigma_i\delta_i)\right) \exp(-\sum_{j=1}^{i-1}\sigma_j\delta_j)\right] L_{i}
$$

Compared with NeRF:

$$
\hat{C}(\mathbf{r})=\sum_{i=1}^{N} T_{i}\left(1-\exp \left(-\sigma_{i} \delta_{i}\right)\right) \mathbf{c}_{i}, \text { where } T_{i}=\exp \left(-\sum_{j=1}^{i-1} \sigma_{j} \delta_{j}\right)
$$

You will find they are very similar.

------------------------------------

## Network Architecture

There're two networks and one is geometry network $\boldsymbol{f}_\varphi$ with 8-layer, 256-width MLP and a single skip connection from the input to the 4th layer. 
<center>
<img src="https://blog-image-zxy.oss-cn-hangzhou.aliyuncs.com/2022-04-19-15-30-57_02aeb7c2.png"/><br>
<div style="color:orange;solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;"><b>The architecture of geometry network</b></div>
</center>
<center>
<img src="https://blog-image-zxy.oss-cn-hangzhou.aliyuncs.com/2022-04-19-15-39-53_421d799f.png"/><br>
<div style="color:orange;solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;"><b>The architecture of radience field network</b></div>
</center>

- VolSDF also uses positional encoding, but one thing we should pay attention is the geometry network only encodes the position $\boldsymbol{x}$ and radiance field network
only encodes view direction $\boldsymbol{v}$. Although the radiance field network also takes in position $\boldsymbol{x}$, it doesn't encode it.

- The geometry network uses softplus as activation function while the radiance 
field network uses ReLU as activation function.

-------------------------------

## Loss Function

Like NeRF, VolSDF uses two network, one for density, and one for color. We can 
define the loss as the distence between rendered color and expected color. But 
this loss is strongly related to the radiance field, and we need add another
loss term to train the geometry network. And we notice that the L2 distence ($d_\Omega(\boldsymbol{x})$) has 
a gradient of length 1. 
So the loss for the geometry network is $\mathbb{E}(||\nabla_x d|| - 1)$

The loss function of VolSDF is:

$$\mathcal{L}(\theta)=\mathcal{L}_{\mathrm{RGB}}(\theta)+\lambda \mathcal{L}_{\mathrm{SDF}}(\varphi), \quad\text{where}$$
$$
\mathcal{L}_{\mathrm{RGB}}(\theta)=\mathbb{E}_{p}\left\|I_{p}-\hat{I}_{\mathcal{S}}\left(\boldsymbol{c}_{p}, \boldsymbol{v}_{p}\right)\right\|_{1}, \quad \text { and } \mathcal{L}_{\mathrm{SDF}}(\varphi)=\mathbb{E}_{\boldsymbol{z}}(\|\nabla d(\boldsymbol{z})\|-1)^{2} \text {, }
$$

$\mathcal{L}_{\mathrm{SDF}}$ is the Eikonal loss, and the samples $\boldsymbol z$ 
are taken to combine a single random uniform space point and a single point 
from $S$ for each pixel $p$. $\lambda$ is a hyper-parameter and set 0.1 in experiments.

Notice that, this loss mainly serve the training the geometry network and not for rendering. So we don't compute the gradient of the function in [Signed Distence Function](#signed-distence-function) , $d$ not $d_\Omega$


--------------------------------------------
## Sampling Algorithm


<center>
<img src="https://blog-image-zxy.oss-cn-hangzhou.aliyuncs.com/2022-04-19-21-23-10_03bbed4c.png" width="50%"/><br>
<div style="color:orange;solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;"><b>Sample Algorithm</b></div>
</center>

The figure above is the procedure of sampling.

1. First, input 2 hyperparameters, one is error threshold $\epsilon$, one is a learnable parameter $\beta$
2. Initialize $\mathcal{T}$ with unifromly sampling 128 points. We can implement by the function `torch.linspace`
3. Initialize $\beta_+$ such that $B_{\mathcal{T},\beta_+}\le\epsilon$ using Lemma 2.

    $$
    B_{\mathcal{T}, \beta} \leq\left(\exp \left(\frac{\alpha}{4 \beta} \sum_{i=1}^{n-1} \delta_{i}^{2}\right)-1\right) \le \epsilon,\ \alpha = \frac{1}{\beta}
    $$
    $$
    \beta \ge \sqrt{\frac{1}{4\log(1+\epsilon)}\sum_{i=1}^{n-1}\delta_i^2}
    $$

4. Sample $\mathcal{T}$, the number of points sampled from each interval is
proportional to its current error bound $\hat{E}$.
    $$
    B_{\mathcal{T}, \beta} = \max _{k \in[n-1]}\left\{\exp \left(-\widehat{R}\left(t_{k}\right)\right)\left(\exp \left(\widehat{E}\left(t_{k+1}\right)\right)-1\right)\right\}
    $$
    $$
    \widehat{R}(t_k)=\sum_{i=1}^{k-1} \delta_{i} \sigma_{i}
    $$
    $$
    \widehat{E}(t_{k+1})=\frac{\alpha}{4 \beta}\left(\sum_{i=1}^{k} \delta_{i}^{2} e^{-\frac{d_{i}^{\star}}{\beta}}\right)
    $$
    $$
    d_{i}^{\star}= \begin{cases}0 & \left|d_{i}\right|+\left|d_{i+1}\right| \leq \delta_{i} \\ \min \left\{\left|d_{i}\right|,\left|d_{i+1}\right|\right\} & \left.|| d_{i}\right|^{2}-\left|d_{i+1}\right|^{2} \mid \geq \delta_{i}^{2} \\ h_{i} & \text { otherwise }\end{cases}
    $$

    One point we should notice is that $d_i, d_{i+1}$ must
    be the same sign, which means the ray in the interval 
    goes throught the boundary and the distence should be zero.
5. If $B_{\mathcal{T},\beta_+} < \epsilon$, then find $\beta_\star$ by binary search 
from $(\beta, \beta_+)$ such that $B_{\mathcal{T},\beta_+} = \epsilon$. The 
max iteration is 5, and then update $\beta_+$.

6. In the end, Estimate $\hat{O}$ using $\mathcal{T}$ and $\beta_+$

The sampling method is based on inverse CDF sampling.