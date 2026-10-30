# IEEE EMBC 2027 研究方案说明文档（Revised Idea 1）

## 论文题目

**中文暂定题目：**  
**去混杂语义解码（Deconfounded Semantic Decoding）：自然阅读脑电中眼动混杂与语言先验控制下的可信语义解码**

**建议英文标题：**  
*Deconfounded Semantic Decoding from Natural-Reading EEG: Does Neural Information Survive Oculomotor and Language-Prior Controls?*

**备选标题：**  
*Are Natural-Reading EEG Decoders Really Decoding Neural Semantics?*

---

# 1. 投稿定位

### 目标会议

IEEE EMBC 2027

### 推荐赛道

- 生物医学信号处理（Biomedical Signal Processing）
- 神经与康复工程（Neural and Rehabilitation Engineering）
- 脑机接口与神经解码（Brain-Computer Interfaces / Neural Decoding）

### 核心论文类型

本研究不再将主要贡献描述为：

> 提出一个新的脑电转文本（EEG-to-Text）生成模型。

而将其定位为：

> **面向自然阅读脑电语义解码的可信性验证与去混杂方法研究。**

核心问题不是：

> 能否得到更高的 BLEU 或 ROUGE 分数？

而是：

> **在显式控制注视相关脑电重叠、眼动动力学以及语言模型先验之后，自然阅读脑电中是否仍然存在可独立恢复的词汇语义信息？**

---

# 2. 研究动机：当前脑电语义解码的可信性问题

近年来脑电转文本研究逐渐从直接的序列到序列生成（Seq2Seq Generation），转向：

- 语义嵌入空间对齐（Semantic Embedding Alignment）；
- 语义检索（Semantic Retrieval）；
- 关键词或语义锚点恢复（Semantic Anchor Recovery）。

然而，这些方法大量依赖自然阅读数据集，例如 ZuCo。

ZuCo 中常见做法是按照：

- 注视起始点（Fixation Onset）；
- 单词注视区间（Fixation-defined Segment）；

对连续脑电进行切分。

这种方法存在三个根本性混杂问题。

---

# 2.1 自然阅读中的注视相关电位重叠

自然阅读中的单次注视通常仅持续数百毫秒。

当当前词的晚期词汇或语义相关神经活动仍在持续时，眼睛可能已经：

- 发起下一次眼跳（Saccade）；
- 注视下一个词；
- 甚至进入第二个后续词。

因此一个简单的注视锁定脑电窗口（Fixation-locked EEG Window）不能直接解释为：

> “当前词对应的独立神经响应”。

连续自然阅读脑电更合理的表达为：

\[
X(t)
=
\sum_k ERP_k(t-t_k)
+
O_{\mathrm{saccade}}(t)
+
O_{\mathrm{EOG}}(t)
+
\epsilon(t)
\]

其中：

- \(ERP_k\)：第 \(k\) 个词相关的事件相关电位；
- \(O_{\mathrm{saccade}}\)：眼跳相关电位；
- \(O_{\mathrm{EOG}}\)：眼电活动；
- \(\epsilon\)：未建模噪声。

因此，本研究不再将：

\[
[t_k+250,\ t_k+550\text{ ms}]
\]

直接定义为：

> 当前词的“纯净 N400 窗口”。

在自然阅读条件下，这种解释并不充分。

---

# 2.2 眼动变量与真实语言加工并非完全可分

首次注视时长（First Fixation Duration, FFD）、凝视时长（Gaze Duration, GD）以及总阅读时间（Total Reading Time, TRT）并不是纯粹的眼动噪声。

它们同时受到以下语言学因素影响：

- 词长（Word Length）；
- 词频（Word Frequency）；
- 可预测性（Predictability）；
- 惊奇度（Surprisal）；
- 词汇加工难度（Lexical Difficulty）。

与此同时，上述语言学变量本身也可能影响脑电中的词汇和语义响应。

因此，不能简单要求：

\[
I(z_{\mathrm{EEG}};\mathrm{FFD})\rightarrow0
\]

并据此宣称得到了“纯神经语义表示”。

这可能会同时删除真实的语言加工信号。

因此，本研究不再将眼动变量简单划分为：

> “纯眼动噪声”与“纯语言信号”。

而采用更加保守的表述：

> **可观测眼动协变量（Observable Oculomotor Covariates）**

并承认这些变量与词汇属性之间可能存在显著共线性。

---

# 2.3 语言模型先验可能掩盖脑电贡献

如果先从脑电预测少量关键词，再输入大型语言模型进行句子生成，即使大型语言模型处于冻结状态（Frozen LLM），仍然不能证明生成内容主要来源于脑电。

原因包括：

1. 预训练语言模型可能接触过目标语料；
2. 少量关键词可能足以触发语言模型自身的文本补全能力；
3. BLEU、ROUGE 或语义相似度的提升可能主要来源于语言模型先验，而不是脑电。

因此，本研究规定：

> **自然语言生成不作为脑电有效性的核心证据。**

论文主结果全部在语言模型介入之前完成。

---

# 3. 核心研究问题

本研究围绕一个总问题展开：

> **自然阅读脑电中是否存在超越眼动结构和语言先验的、可独立恢复的词汇语义信息？**

进一步拆分为四个研究问题。

### RQ1：重叠问题

自然阅读脑电中的语义解码性能，有多少依赖于连续注视和眼跳造成的时间重叠结构？

### RQ2：眼动混杂问题

模型学习到的脑电表示中，有多少信息可以由眼电、眼跳和注视行为解释？

### RQ3：真实神经贡献

经过眼动残差化处理后，真实脑电是否仍然显著优于：

- 高斯噪声（Gaussian Noise）；
- 时间打乱脑电（Temporally Shuffled EEG）；
- 纯眼电/眼动输入（EOG/Gaze-only）；
- 词频先验（Frequency Prior）？

### RQ4：预处理有效性

如果眼动残差化后解码性能下降，这种下降究竟来自：

- 真正去除了眼动混杂；

还是：

- 预处理方法误删了真实神经信号？

---

# 4. 总体框架

整体流程定义为：

\[
\boxed{
\text{连续脑电 + 眼电 + 眼动数据}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{眼动协变量建模与残差化}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{词级脑电切片}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{轻量脑电语义编码器}
}
\]

\[
\downarrow
\]

\[
\boxed{
\text{固定候选集词级语义检索}
}
\]

大型语言模型仅作为可选的下游展示，不参与核心结论建立。

---

# 5. 模块一：眼动残差化脑电构建

## 5.1 不再使用“反卷积脑电”这一表述

反卷积回归（Deconvolution Regression）或回归事件相关电位分析（Regression-based ERP, rERP）本身主要估计各事件和协变量对应的回归系数：

\[
\beta(t)
\]

它不会直接输出：

> 每个词对应的一段“纯净脑电”。

因此，本研究不再使用：

> “Deconvolved EEG”

这一容易引起误解的术语。

统一改为：

> **眼动残差化脑电（Oculomotor-Residualized EEG）**

---

# 5.2 眼动残差化定义

连续脑电记为：

\[
X_{\mathrm{EEG}}
\]

使用仅由眼动与眼电信息构建的设计矩阵：

\[
X_{\mathrm{ocul}}
\]

拟合眼动相关成分：

\[
\hat X_{\mathrm{ocul}}
=
X_{\mathrm{ocul}}
\hat\beta_{\mathrm{ocul}}
\]

最终得到：

\[
X_{\mathrm{res}}
=
X_{\mathrm{EEG}}
-
\hat X_{\mathrm{ocul}}
\]

其中：

\[
X_{\mathrm{res}}
\]

定义为：

> **眼动残差化连续脑电（Oculomotor-Residualized Continuous EEG）**

之后再按照注视起始点切分词级脑电。

---

# 5.3 解码输入中禁止使用文本协变量

这是本研究最严格的数据泄露控制之一。

用于生成解码输入的设计矩阵中，只允许包含不依赖目标文本语义的可观测眼动和眼电变量。

例如：

- 水平眼电（Horizontal EOG）；
- 垂直眼电（Vertical EOG）；
- 眨眼（Blink）；
- 入射眼跳幅度（Incoming Saccade Amplitude）；
- 出射眼跳幅度（Outgoing Saccade Amplitude）；
- 眼跳方向（Saccade Direction）；
- 注视位置（Fixation Position）；
- 注视时间结构（Fixation Timing）。

以下变量：

- 词长；
- 词频；
- 惊奇度；
- 可预测性；
- 词性；

**禁止用于构造眼动残差化脑电。**

尤其禁止：

\[
X_{\mathrm{lex}}\beta_{\mathrm{lex}}
\]

直接或间接参与测试集 EEG 输入构建。

---

# 5.4 词汇协变量的角色

词汇变量仅用于：

> **神经生理诊断分析（Neurophysiological Diagnostic Analysis）**

例如在独立的回归事件相关电位分析中研究：

\[
EEG
\sim
\text{Word Length}
+
\text{Frequency}
+
\text{Surprisal}
+
\text{Saccade}
+
\text{Fixation}
\]

该分析只用于：

- 检查晚期语言相关响应；
- 检查协变量贡献；
- 验证统计模型合理性。

其结果绝不作为语义解码器输入。

---

# 6. 眼动协变量的共线性分析

眼动变量并不是完全独立于词汇变量的。

例如：

\[
\text{Word Length}
\rightarrow
\text{Landing Position}
\]

以及：

\[
\text{Word Length}
\rightarrow
\text{Saccade Amplitude}
\]

因此眼动残差化可能间接删除部分与语言因素相关的共享方差。

本研究不会把残差化结果解释为：

> “纯神经信号”。

而采用更加保守的表述：

> **降低显式可观测眼动结构后保留下来的脑电信号。**

为量化变量间依赖关系，报告：

- 皮尔逊相关系数（Pearson Correlation）；
- 斯皮尔曼相关系数（Spearman Correlation）；
- 方差膨胀因子（Variance Inflation Factor, VIF）；
- 设计矩阵条件数（Condition Number）。

重点分析：

- 词长 vs. 眼跳幅度；
- 词长 vs. 注视位置；
- 词频 vs. FFD；
- 惊奇度 vs. GD/TRT；
- FFD vs. 眼跳幅度。

---

# 7. 模块二：词级脑电语义编码器

眼动残差化后得到连续信号：

\[
X_{\mathrm{res}}
\]

随后按照每个目标词的注视起始点构建固定规则的词级脑电片段：

\[
E_k
=
X_{\mathrm{res}}
[t_k-\Delta_1,\ t_k+\Delta_2]
\]

所有方法必须使用完全一致的时间范围。

然后输入轻量脑电编码器：

\[
z_k^{EEG}
=
f_\theta(E_k)
\]

推荐使用简单模型：

- 一维时域卷积（1D Temporal Convolution）；
- 轻量状态空间模型（Small State Space Model）；
- 通道注意力（Channel Attention）；
- 投影多层感知机（Projection MLP）。

本研究避免堆叠：

- 情感预测；
- 主题预测；
- 惊奇度预测；
- 多任务属性头；
- 大型生成式解码器。

原因是：

> 本研究的主要变量应该是“输入中是否包含眼动混杂”，而不是模型结构复杂度。

---

# 8. 任务定义：词级语义检索

本研究固定采用：

> **词级语义检索（Word-level Semantic Retrieval）**

而不混合：

- 词级检索；
- 句级检索；
- 自由句子生成。

对于第 \(k\) 个目标词：

\[
w_k
\]

由冻结文本编码器得到词语义嵌入：

\[
z_k^{text}
=
g(w_k)
\]

脑电编码器输出：

\[
z_k^{EEG}
\]

使用对比学习损失（Contrastive Loss）进行训练：

\[
\mathcal L_{\mathrm{sem}}
=
-\log
\frac{
\exp(
\mathrm{sim}(z_k^{EEG},z_k^{text})/\tau
)
}{
\sum_j
\exp(
\mathrm{sim}(z_k^{EEG},z_j^{text})/\tau
)
}
\]

其中：

\[
\tau
\]

为温度系数（Temperature）。

---

# 9. 目标词过滤规则

主实验仅保留实词（Content Words）。

包括：

- 名词（Noun）；
- 动词（Verb）；
- 形容词（Adjective）；
- 可选：副词（Adverb）。

排除：

- 标点符号；
- 停用词（Stop Words）；
- 极低频词；
- 无有效注视的词；
- EEG/眼动数据质量不合格的词。

词性过滤与词频过滤分开定义。

例如：

\[
POS
\in
\{\mathrm{NOUN,VERB,ADJ,ADV}\}
\]

同时：

\[
Frequency_{\mathrm{train}}
\ge m
\]

其中阈值 \(m\) 仅根据训练集确定。

---

# 10. 固定候选词库设计

语义检索采用固定大小候选词库（Fixed Candidate Gallery）。

对于目标词：

\[
w^+
\]

构建：

\[
G
=
\{
w^+,
w_1^-,
\dots,
w_{N-1}^-
\}
\]

所有实验固定：

\[
|G|=N
\]

推荐主实验：

\[
N=100
\]

必要时增加：

\[
N=250
\]

作为规模敏感性实验。

所有输入条件和模型必须使用完全一致的候选集。

---

# 11. 候选词库控制条件

设置两个主要检索条件。

## 11.1 随机候选词库

负样本从符合过滤规则的实词中随机选择。

该任务用于衡量基本语义可分性。

---

## 11.2 属性匹配候选词库

为了避免模型仅依赖：

- 词长；
- 词频；
- 词性；

完成检索，对每个目标词构建匹配负样本。

负样本应尽量匹配：

- 词长（Word Length）；
- 对数词频（Log Word Frequency）；
- 词性（Part-of-Speech）。

必要时还可匹配：

- 平均注视时长；
- 字符数；
- 子词数量。

该设置定义为：

> **属性匹配语义检索（Attribute-Matched Semantic Retrieval）**

它将作为论文最严格的主结果之一。

---

# 12. 检索评价指标

主要指标：

- 前 K 项召回率（Recall@K）；
- 平均倒数排名（Mean Reciprocal Rank, MRR）；
- 归一化折损累计增益（Normalized Discounted Cumulative Gain, nDCG）。

固定候选库大小后必须同时报告随机基线：

\[
Recall@K_{\mathrm{chance}}
=
\frac{K}{N}
\]

例如：

\[
N=100
\]

时：

\[
Recall@5_{\mathrm{chance}}
=
5\%
\]

---

# 13. 未见文本设置的定义

主实验采用：

> **未见被试 + 未见文本（Unseen Subject + Unseen Text）**

测试阶段目标词允许出现在候选词库中，因为候选词库本质上相当于测试词典。

但是：

> 测试 EEG 与目标词之间的配对关系从未在训练阶段出现。

因此该任务更准确地描述为：

> **未见 EEG-词语配对条件下的开放词表语义嵌入检索（Open-vocabulary Embedding Retrieval under Unseen EEG-Word Pairs）**

而不是传统封闭词表分类。

---

# 14. 模块三：独立眼动探针分析

本研究删除原方案中的：

\[
Z_n \perp Z_o
\]

潜变量正交分支。

原因是：

- \(Z_o\) 定义不明确；
- 线性去相关并不代表统计独立；
- 会额外引入难以解释的模型组件。

因此眼动残留信息通过训练后独立探针（Post-hoc Probe）进行测量。

---

# 15. 独立探针设置

脑电编码器训练结束后完全冻结。

重新训练：

### 线性探针（Linear Probe）

以及：

### 非线性多层感知机探针（Non-linear MLP Probe）

从：

\[
z^{EEG}
\]

预测：

- 眼电幅度；
- 眼跳幅度；
- 注视位置；
- 眨眼状态；
- FFD；
- GD。

探针模型不参与原始脑电表示学习。

因此探针结果属于独立诊断，而不是训练目标本身。

报告：

- 决定系数（\(R^2\)）；
- 平均绝对误差（MAE）；
- 如为分类变量则报告 AUROC。

---

# 16. 数据集设计

## 16.1 ZuCo 1.0 / 2.0

ZuCo 用于：

> **自然阅读条件下的眼动混杂与语义解码主实验。**

主要利用：

- 连续脑电；
- 同步眼动；
- 注视时间；
- 眼跳信息；
- 单词刺激标签。

ZuCo 不再承担：

> “纯净 N400 验证”

这一角色。

---

## 16.2 DERCo

DERCo 用于：

> **受控条件下的时间响应与正向神经生理验证。**

其主要作用不是直接作为 ZuCo 的训练扩展，而是验证：

- 晚期词汇效应是否具有合理时间结构；
- 预处理是否保留已知语言相关脑电响应。

因此：

\[
\text{DERCo}
\rightarrow
\text{受控时间验证}
\]

而：

\[
\text{ZuCo}
\rightarrow
\text{自然阅读去混杂解码}
\]

两者承担不同科学角色。

---

# 17. 实验一：眼动残差化是否改变语义解码？

固定：

- EEG 编码器；
- 文本编码器；
- 训练目标；
- 候选词库；
- 数据划分。

仅改变脑电输入构造方式。

| 条件 | 描述 |
|---|---|
| 原始注视脑电（Raw Fixation EEG） | 按注视起点直接切片 |
| 常规去伪迹脑电（ICA-cleaned EEG） | 常规 ICA/EOG 处理后切片 |
| 眼动残差化脑电（Oculomotor-Residualized EEG） | 从连续 EEG 中减去眼动协变量拟合成分后切片 |

核心问题：

> 控制眼动结构后，词级语义检索性能是否仍然存在？

---

# 18. 实验二：信号有效性对照

所有模型必须同时测试：

### 1. 真实脑电（Real EEG）

\[
X_{\mathrm{real}}
\]

### 2. 高斯噪声（Gaussian Noise）

\[
X_{\mathrm{noise}}
\sim
\mathcal N(0,I)
\]

### 3. 时间打乱脑电（Temporally Shuffled EEG）

保留总体信号分布或频谱结构，但破坏：

\[
EEG
\leftrightarrow
Word
\]

时间对应关系。

### 4. 纯眼电/眼动条件（EOG/Gaze-only）

仅输入：

- EOG；
- 注视时长；
- 眼跳参数；
- 注视位置。

目的不是预设：

\[
Real > EOG > Noise
\]

而是直接测量各种输入中可恢复的信息量。

---

# 19. 实验三：合成信号正向对照

这是本研究的关键质量控制实验。

为了防止出现：

> “性能下降只是因为残差化算法把所有信号都删掉了”

在真实连续脑电中人工注入已知的词事件相关信号：

\[
s_k(t)
=
A_k h(t-t_k)
\]

其中：

- \(h(t)\)：预先设定的晚期平滑波形；
- \(A_k\)：已知注入幅度；
- \(t_k\)：目标词事件时间。

构建：

\[
X_{\mathrm{inj}}
=
X_{\mathrm{EEG}}
+
s(t)
\]

之后运行完整的眼动残差化流程。

检查：

1. 波形时间位置是否能够恢复；
2. 注入幅度 \(A_k\) 是否能够恢复；
3. 不同信噪比下恢复能力是否稳定。

推荐设置：

\[
SNR
\in
\{-10,-5,0,+5\text{ dB}\}
\]

核心判断：

> 眼动残差化流程在降低眼动结构的同时，仍能够保留已知时间尺度和幅度的事件相关神经信号。

---

# 20. 实验四：真实数据正向对照

除合成实验外，还应验证真实脑电中的基本神经结构没有被整体破坏。

可报告：

- ERP/FRP 波形形态；
- 信号方差保留率；
- 通道间结构；
- 受控数据集上的已知词汇调制；
- 不同重复试次间可靠性。

这里不将：

> “某个词频效应一定必须存在”

作为绝对标准。

重点是证明：

> 残差化不是简单把连续 EEG 压缩成接近零或极低方差信号。

---

# 21. 实验五：已发表模型外部验证

如果论文希望讨论：

> 既往脑电解码结果可能部分受到眼动结构影响。

仅使用自建编码器是不够的。

因此至少复现一个已发表模型。

优先选择：

- DeWave；
- 或一个公开代码稳定、能够替换输入特征的已发表脑电语义检索/脑电转文本模型。

对同一个已发表模型分别测试：

\[
\text{Published Model + Raw EEG}
\]

与：

\[
\text{Published Model + Residualized EEG}
\]

并加入：

\[
\text{Noise / EOG Control}
\]

如果：

\[
\Delta_{\mathrm{Raw\rightarrow Residualized}}
\]

在自建简单模型和已发表模型上均出现类似趋势，则结论具有更强外部有效性。

---

# 22. 实验六：独立眼动探针

比较：

\[
z_{\mathrm{raw}}
\]

与：

\[
z_{\mathrm{res}}
\]

在独立线性探针与非线性探针下预测：

- EOG；
- saccade amplitude；
- fixation position；
- FFD；
- GD。

理想情况为：

\[
Probe(z_{\mathrm{res}})
<
Probe(z_{\mathrm{raw}})
\]

同时：

\[
SemanticRetrieval(z_{\mathrm{res}})
>
Chance
\]

但论文不会声称：

> 表示已经与眼动统计独立。

---

# 23. 实验七：候选词属性控制

对每个测试目标词分别测试：

### 随机候选库

以及：

### 词长/词频/词性匹配候选库

如果模型在严格匹配条件下仍显著超过随机水平，则更能支持：

> 模型并非只根据长度、频率或词性结构完成检索。

这是脑电语义解码可信性的重要证据。

---

# 24. 泄露控制协议

所有以下操作均只能基于训练集：

- 标准化参数；
- PCA；
- 词频阈值；
- vocabulary filtering；
- 数据质量阈值；
- 模型选择；
- 超参数选择。

用于脑电解码输入构造的残差化过程：

> 不允许使用目标词文本属性。

测试阶段如进行被试内眼动拟合，只能使用：

- EEG；
- EOG；
- gaze；

等无文本标签信息。

不能使用：

- word identity；
- frequency；
- surprisal；
- semantic embedding。

---

# 25. 主要评价指标

## 25.1 语义解码

- Recall@1；
- Recall@5；
- Recall@10；
- MRR；
- nDCG。

---

## 25.2 混杂诊断

- EOG probe \(R^2\)；
- saccade probe \(R^2\)；
- fixation probe \(R^2\)；
- EOG/gaze-only retrieval performance。

---

## 25.3 预处理有效性

- 合成信号幅度恢复相关系数；
- 合成信号波形恢复误差；
- ERP/FRP 方差保留率；
- 处理前后信号功率变化。

---

## 25.4 统计检验

采用：

- 配对自助重采样（Paired Bootstrap）；
- 置换检验（Permutation Test）；
- 被试级置信区间（Subject-level Confidence Interval）。

---

# 26. 核心消融实验

控制在 EMBC 四页正文可以容纳的规模。

### A1：输入构造

原始注视 EEG  
vs.  
ICA 去伪迹 EEG  
vs.  
眼动残差化 EEG

### A2：输入真实性

真实 EEG  
vs.  
EOG/Gaze-only  
vs.  
Shuffle  
vs.  
Noise

### A3：检索难度

随机候选词库  
vs.  
词长/词频/词性匹配候选词库

### A4：模型外部有效性

自建轻量编码器  
vs.  
已发表模型

---

# 27. 可能结果与解释边界

## Outcome A

若：

\[
Residualized\ EEG
>
Matched\ Baseline
>
Chance
\]

并且：

\[
OcularProbe_{\mathrm{res}}
<
OcularProbe_{\mathrm{raw}}
\]

同时合成信号能够稳定恢复，则支持：

> 自然阅读脑电在降低显式眼动结构后，仍包含可恢复的词汇语义信息。

---

## Outcome B

若：

\[
Raw
\gg
Residualized
\approx
Chance
\]

不能立即解释为：

> 过去模型完全依赖眼动混杂。

只有在同时满足：

1. 合成信号正向对照证明预处理保留已知神经信号；
2. 已发表模型也出现类似性能下降；
3. 眼动 probe 显著下降；

之后，才可以更谨慎地表述为：

> 原始注视切片中的部分语义可解码性可能来自可观测眼动结构。

---

## Outcome C

若：

\[
Raw
>
Residualized
>
Chance
\]

这可能是最合理的结果。

说明：

> 自然阅读脑电中确实存在词汇语义信息，但原始注视脑电的表观性能可能同时包含眼动相关贡献。

---

# 28. 预期论文贡献

本文不宣称首次实现：

- EEG semantic retrieval；
- EEG-text contrastive learning；
- semantic anchor decoding。

主要贡献定义为以下四点。

### Contribution 1

提出一种严格无文本标签注入的：

> **眼动残差化自然阅读脑电构建流程**

并明确区分解码输入预处理与词汇协变量诊断分析。

### Contribution 2

通过：

- 合成信号正向对照；
- 真实神经信号保留分析；

验证眼动残差化不会简单破坏所有晚期事件相关脑电结构。

### Contribution 3

通过：

- EOG/gaze-only；
- Gaussian noise；
- temporal shuffle；
- attribute-matched gallery；
- post-hoc ocular probes；

系统评估自然阅读脑电语义解码中的非神经混杂。

### Contribution 4

在自建轻量模型之外，使用至少一个已发表脑电解码模型重复主要实验，提高结论的外部有效性。

---

# 29. EMBC 4+1 页布局

## Page 1：Introduction

重点讲：

- 自然阅读脑电语义解码；
- 注视相关电位重叠；
- 眼动与语言变量共线；
- 现有 retrieval/generation 指标无法区分神经与眼动贡献。

核心问题：

> Does semantic decodability survive explicit oculomotor control?

---

## Page 2：Method

Figure 1：

**连续 EEG/EOG/Gaze → 眼动残差化 → 词级 epoch → 语义编码器 → 固定候选词检索**

同时画出两条独立分析路径：

### 解码路径

不使用文本协变量。

### 诊断路径

使用词频、词长、surprisal 等变量进行 rERP 分析。

---

## Page 3：Main Results

Table I：

| 模型/输入 | Raw | ICA | Residualized | EOG | Shuffle | Noise |
|---|---:|---:|---:|---:|---:|---:|
| 自建编码器 | | | | | | |
| 已发表模型 | | | | | | |

主指标：

- Recall@5；
- MRR。

---

## Page 4：Validity Analysis

Figure 2：

左侧：

> Synthetic signal recovery vs. SNR

右侧：

> Semantic retrieval vs. ocular probe performance

Table II：

随机候选库 vs. 属性匹配候选库。

最后 Conclusion。

---

## Page 5

- Acknowledgments
- AI Tool Disclosure
- References

---

# 30. 最小可行版本（MVP）

第一阶段只实现：

1. ZuCo 一个版本；
2. 原始注视 EEG；
3. 眼动残差化 EEG；
4. 简单 EEG 编码器；
5. 100-word 固定候选库；
6. 随机候选库；
7. 词长/词频/词性匹配候选库；
8. Real / EOG / Shuffle / Noise；
9. 线性与 MLP 独立眼动 probe；
10. synthetic late-effect injection recovery。

如果这些结果已经形成稳定趋势，再加入：

11. 一个已发表模型；
12. DERCo 时间验证。

大型语言模型重构不属于 MVP。

---

# 31. 一句话论文定位

> **本研究不再尝试提出另一个更复杂的 EEG-to-Text 解码器，而是检验自然阅读脑电中报告的词汇语义可解码性，在显式控制眼动结构、候选词属性和语言先验之后是否仍然成立。**

英文摘要式表达：

> *Rather than proposing another EEG-to-text decoder, we test whether lexical-semantic decodability in natural-reading EEG survives explicit control of oculomotor structure, candidate-word attributes, and non-neural decoding priors.*

---

# 32. 当前最关键的实验判据

整篇论文的核心不再是：

\[
\text{Our Model} > \text{SOTA}
\]

而是：

\[
\boxed{
\text{Residualized Real EEG}
>
\text{Matched Controls}
>
\text{Chance}
}
\]

同时满足：

\[
\boxed{
\text{Synthetic Neural Signal Survives Residualization}
}
\]

以及：

\[
\boxed{
\text{Decodable Oculomotor Information Decreases}
}
\]

只有这三个条件同时成立，才能较有说服力地证明：

> **模型恢复的不是单纯的眼动结构，而是至少包含独立于显式眼动协变量之外的神经语言信息。**