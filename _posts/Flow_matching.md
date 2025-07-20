---
title: 'flow matching'
date: 2025-07-20
permalink: /posts/2025/07/20
tags:
  - flow
  - generative model

---

## 一.背景介绍

flow matching 本质是一种流映射关系，建立一个源分布与目标分布的映射模型，具体是通过学习概率路径$p_t(x)$和速度场（向量场）$v_t(x)$来链接源分布于目标分布, 流flow表示为$\phi_t(x)$,他们之间的关系可以表示为：
$$
\frac{d\phi_t(x)}{dt} = \mathbf{v}(\phi_t(x), t)
$$
所以flow matching的目标为学习一个向量场$v_t(x)$，使得他可以生成流$\phi_t(x)$，流上的每一个点为路径$p_t(x)$，并且满足分布$p_0=p(x_0), p_1=q(x_1)$,时间$t$通常为0到1,其中他们都必须满足连续性方程。

![method](https://huangfan0.github.io/images/2-1.png)

## 二. continuous flow matching

1.Continuous Normalizing Flow(CNF)学习的目标函数为速度场，但是真实的目标分布未知，所以论文证明带条件的速度场与目标速度场等价，即：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t,X_t} \left\| u_t^\theta(X_t) - u_t(X_t) \right\|^2, \text{ where } t \sim \mathcal{U}[0,1] \text{ and } X_t \sim p_t,
$$
$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,q(x_1),p_t(x|x_1)} \left\| v_t(x) - u_t(x|x_1) \right\|^2,
$$
$$\nabla_{\theta} \mathcal{L}_{\text{FM}}(\theta) = \nabla_{\theta} \mathcal{L}_{\text{CFM}}(\theta)$$

2.最优传输速度场
条件需要满足原目标分布为高斯分布，$p1$均值为$X_1$,方差为一个很小的数值$\sigma_t(x_1)$,,因此，$\mu_t(x) = t x_1, \text{ and } \sigma_t(x) = 1 - (1 - \sigma_{\min})t.$

此时目标函数可以简化为：
$$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min})x}{1 - (1 - \sigma_{\min})t},u_t(x|x_1) \approx x_1 - x_0$$
此时是匀速直线运动

3.Mean Flows for One-step Generative Modeling

原来的目标函数为瞬时速度，mean flow 将其优化为平均速度，修改了损失函数,最后的效果是可以一次采样
$$u(z_t, r, t) = \underbrace{v(z_t, t)}_{\text{instant. vel.}} - (t - r) \underbrace{\frac{d}{dt}u(z_t, r, t)}_{\text{time derivative}}$$

$$\mathcal{L}(\theta) = \mathbb{E} \left\| u_{\theta}(z_t, r, t) - \text{sg}(u_{\text{tgt}}) \right\|_2^2,

\text{where} \quad u_{\text{tgt}} = v(z_t, t) - (t - r) \left( v(z_t, t) \partial_z u_{\theta} + \partial_t u_{\theta} \right),$$
3.最简单的CNF流程

① 线性插值采样  $X_t = tX_1 + (1 - t)X_0 \sim p_t.$

② 神经网络拟合学习速度场，学习目标函数

③ ODE采样，可以使用一阶欧拉法，或二阶龙格 - 库塔法。

他们之间的关系可以表示为：

![method](https://huangfan0.github.io/images/2-2.png)

## 三. discrete flow matching

1.离散的源分布可以使用masked token， uniform distribution 来等价连续的flow中的源分布$p$

2.插值的使用delta function 来取值，通过一个调度器$\kappa_t$来表示$x_t$取值的$x_0$或$x_1$的概率
$$p_t(x^i|x_0, x_1) = (1 - \kappa_t) \delta_{x_0}(x^i) + \kappa_t \delta_{x_1}(x^i)$$
$$\delta(x, z) = 
  \begin{cases} 
   1 & \text{if } x = z, \\
   0 & \text{otherwise}.
  \end{cases}
$$

3.离散的flow习惯学习的路径，通过路径可以推断速度然后采样

路径的表示：$p_{t|0,1}^i(x^i|x_0, x_1) = \kappa_t \delta(x^i, x_1^i) + (1 - \kappa_t) \delta(x^i, x_0^i),$

速度的表示：$ u_t^i(y^i, x^i | x_1) = \frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ \delta(y^i, x_1^i) - \delta(y^i, x^i) \right] $ 

4.损失函数的学习
$$
\ell_i(x_1, x_t, t) = -\frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ p_{1|t}(x_t^i | x_t) - \delta_{x_1^i}(x_t^i) + (1 - \delta_{x_1^i}(x_t^i)) \log p_{1|t}(x_1^i | x_t) \right]
$$
状态保持强化（当 ($x_t^i = x_1^i$）)

$   \text{loss}_{\text{keep}} = -\frac{\dot{\kappa}_t}{1 - \kappa_t} \left[ p_{1|t}(x_t^i | x_t) - 1 \right]$

   - 当状态未变化时，最大化保持当前状态的概率
   - $(p_{1|t}(x_t^i | x_t) \rightarrow 1)$ 时损失最小化

状态转移优化（当 ($x_t^i \neq x_1^i$)）

$   \text{loss}_{\text{transfer}} = -\frac{\dot{\kappa}_t}{1 - \kappa_t} \log p_{1|t}(x_1^i | x_t)
$
   - 当状态应变化时，最大化目标状态的对数似然
   - 本质是负对数似然损失（NLL）

5.采样
facebook 的代码更多的是计算出$p_{1|t}^i(\cdot|X_t)$后，利用多项式采样每一个token的值
$$\begin{align*}
    & X_1^i \sim p_{1|t}^i(\cdot|X_t)\\
    & \lambda^i \gets \sum_{x^i \ne X_t^i} u_t^i(x^i, X_t^i|X_1^i)\\
    & Z^i_{\text{change}} \sim U[0,1]\\
    & X_{t+h}^i \sim 
    \begin{cases}
        \frac{u_t^i(\cdot, X_t^i|X_1^i)}{\lambda^i}(1-\delta_{X_t^i}(\cdot)) & \text{if } Z^i_{\text{change}} \le 1-e^{-h\lambda^i}\\
        \delta_{X_t^i}(\cdot) & \text{else}
    \end{cases}
\end{align*}$$

$$ u_t^i(x^i, y^i|x_1^i) = \hat{u}_t^i(x^i, y^i|x_1^i) + c_{\text{div\_free}}\left[\hat{u}_t^i(x^i, y^i|x_1^i) - \check{u}_t^i(x^i, y^i|x_1^i) \right],$$