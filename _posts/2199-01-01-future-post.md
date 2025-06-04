---
title: 'Generative Recommendation with Continuous-Token Diffusion'
date: 2025-06-03
permalink: /posts/2025/06/03
tags:
  - GNN
  - Recommendation
  - llm
  - diffusion
---
## 背景
大模型推荐范式：

(1)  user&item tokenization：输入信息包括 User,item 变为tokenization，和文本信息的prompt交互

(2)recommendation generation：大模型推理建模用户的偏好，使用自然语义或输出token产生合适的推荐

其中user，item变为token这一步骤可以分为离散的和连续的两种方法，论文任务离散的tokenization会产生信息压缩和幻觉（information compression和hallucination）
具体局限包括：（1）模型的性能常常受有损标记局限（lossy tokenization）因为在离散的过程中，信息通常被压缩，所以需要一些额外的离散对齐方法；
（2）在现实世界的推荐场景中，大量用户和项目的标记化和生成被限制在有限的离散标记词汇表中，所以难以覆盖海量的用户或物品。


所以论文想要从continuous representation的角度生成连续的token。论文使用diffusion来解决连续的表示。

## 方法
<!-- ![method](https://github.com/huangfan0/huangfan0.github.io/raw/master/images/1-1.png "method")
![method](https://github.com/huangfan0/huangfan0.github.io/raw/master/images/1-1.png) -->

![method](https://huangfan0.github.io/images/1-1.png)

<!-- ![method](https://raw.githubusercontent.com/huangfan0/huangfan0.github.io/master/images/1-1.png) -->


整体流程为：1交互矩阵通过GNN编码得到初步的协同过滤的向量表示

2.Additive Continuous Tokenizer：使用masking operation和K-way  architecture 把第一步中的向量变为连续向量

最终输入LLM的格式：物品标题 + <连续token_1> <连续token_2> ...
 例："Apple Vision Pro [0.12, -0.3] [0.8, 0.1]"


3.Generative Recommendion with Continuous-Token Diffusion,

将第二部中得到的连续向量编码和大模型的prompt作为输入，大模型输出用户的偏好c
(为什么不用大模型直接输出推荐结果？可能是大模型擅长语义推理，但不擅长连续空间的精准推荐，所以解释偏好的描述，提供高层的语义指导如商品的品牌，后面的diffusion才是将语义指导转化为精准的推荐信号)

4.Contrastive Denoising Diffusion

  • 将第一步中的向量输入diffusion学习数据分布，这里除了正样本的向量还有负样本的向量。
  • 偏好C是在反向过程解噪时与噪声和步长一起作为输入，输入是原来的初始向量x0。所以这里的向量增强表示除了有推荐的协同信息，也融合了大模型输入的用户偏好信息。
  • 除了正常的推荐BPR损失，diffusion损失，正负样本的噪声预测误差也会有BPR损失。

5 Hybrid Item Retriever

Top-K计算复杂，使用向量相似性计算
$s_{ij}=\dfrac{\mathbf{y}_{i}\mathbf{q}_{j}}{\|\mathbf{y}_{i}\|\|\mathbf{q}_{j}\|}\cdot(1+\pi)$，当大模型输出偏好$Z_i$符合商品时，$\pi$是是一个很小的常数，否则为0


## 实验结果
数据集： Software , Beauty and LastFM

  • 有大模型微调的推荐系统比传统的方法性能效果要好

  • 表示user，item连续token的模型（CoLLM,LlaRA）比离散的模型（P5）要好

  • GNN-based 协同过滤比传统方法要好 

![result](https://huangfan0.github.io/images/1-2.png)










<!-- ---
title: 'Blog Post number 4'
date: 2015-08-14
permalink: /posts/2012/08/blog-post-4/

permalink: /posts/2012/08/blog-post-4/

tags:
  - cool posts
  - category1
  - category2
---

This is a sample blog post. Lorem ipsum I can't remember the rest of lorem ipsum and don't have an internet connection right now. Testing testing testing this blog post. Blog posts are cool.

Headings are cool
======

You can have many headings
======

Aren't headings cool?
------ -->