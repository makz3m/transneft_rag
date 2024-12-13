# Attention Is All You Need

**Authors:**
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

**Affiliations:**
Google Brain, Google Research, University of Toronto

**Abstract:**
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

**Contributions:**
- Jakob proposed replacing RNNs with self-attention and initiated the evaluation of this idea.
- Ashish, along with Illia, designed and implemented the first Transformer models and was crucially involved in all aspects of the work.
- Noam proposed scaled dot-product attention, multi-head attention, and the parameter-free position representation, contributing to nearly every detail.
- Niki designed, implemented, tuned, and evaluated numerous model variants in the original codebase and tensor2tensor.
- Llion experimented with novel model variants, managed the initial codebase, and worked on efficient inference and visualizations.
- Lukasz and Aidan dedicated extensive time to designing and implementing various parts of tensor2tensor, significantly improving results and accelerating research.

**Conference:**
31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.# Introduction

Recurrent neural networks, particularly long short-term memory (LSTM) and gated recurrent neural networks (GRNN), have been established as state-of-the-art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have continued to push the boundaries of recurrent language models and encoder-decoder architectures.

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states \( h_t \), as a function of the previous hidden state \( h_{t-1} \) and the input for position \( t \). This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks and conditional computation, while also improving model performance in the latter case. However, the fundamental constraint of sequential computation remains.

Attention mechanisms have become integral to compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. In most cases, however, such attention mechanisms are used in conjunction with a recurrent network.

In this work, we propose the Transformer, a model architecture that eschews recurrence and instead relies entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can achieve a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

# Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet, and ConvS2S, all of which use convolutional neural networks as basic building blocks, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions. In the Transformer, this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect counteracted with Multi-Head Attention.

Self-attention, sometimes called intra-attention, is an attention mechanism relating different positions of a single sequence to compute a representation of the sequence. Self-attention has been successfully used in various tasks including reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representations. End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks.

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention, and discuss its advantages over other models.

# Model Architecture

Most competitive neural sequence transduction models have an encoder-decoder structure. Here, the encoder maps an input sequence of symbol representations \( (x_1, ..., x_n) \) to a sequence of continuous representations \( z = (z_1, ..., z_n) \). Given \( z \), the decoder then generates an output sequence \( (y_1, ..., y_m) \) of symbols one element at a time. At each step, the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.The Transformer follows a specific architecture that utilizes stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, as illustrated in Figure 1.

### 3.1 Encoder and Decoder Stacks

**Encoder:** The encoder consists of a stack of \( N = 6 \) identical layers. Each layer contains two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. A residual connection is employed around each of the two sub-layers, followed by layer normalization. Specifically, the output of each sub-layer is computed as \( \text{LayerNorm}(x + \text{Sublayer}(x)) \), where \( \text{Sublayer}(x) \) represents the function executed by the sub-layer. To support these residual connections, all sub-layers in the model, including the embedding layers, produce outputs with a dimension of \( d_{\text{model}} = 512 \).

**Decoder:** The decoder also comprises a stack of \( N = 6 \) identical layers. In addition to the two sub-layers present in each encoder layer, the decoder includes a third sub-layer that performs multi-head attention over the output of the encoder stack. Similar to the encoder, residual connections are utilized around each of the sub-layers, followed by layer normalization. The self-attention sub-layer in the decoder stack is modified to prevent positions from attending to subsequent positions. This masking, along with the fact that the output embeddings are shifted by one position, ensures that the predictions for position \( i \) can only depend on the known outputs at positions less than \( i \).

### 3.2 Attention

An attention function can be defined as a mapping from a query and a set of key-value pairs to an output, where the query, keys, values, and output are all represented as vectors. The output is calculated as a weighted sum of the values, with the weights determined by the compatibility of the query with the corresponding keys.The text discusses two key concepts in the field of attention mechanisms used in neural networks: Scaled Dot-Product Attention and Multi-Head Attention.

### Scaled Dot-Product Attention
Scaled Dot-Product Attention is a specific type of attention mechanism where the input consists of queries and keys of dimension \(d_k\), and values of dimension \(d_v\). The process involves the following steps:
1. Compute the dot products of the query with all keys.
2. Scale the dot products by dividing by \(\sqrt{d_k}\).
3. Apply a softmax function to obtain weights for the values.

The attention function can be expressed mathematically as:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]
This method is efficient and can be computed in parallel for a set of queries packed into a matrix \(Q\), with keys and values also packed into matrices \(K\) and \(V\).

The text contrasts two commonly used attention functions: additive attention and dot-product attention. While both have similar theoretical complexity, dot-product attention is preferred in practice due to its speed and space efficiency. However, for larger values of \(d_k\), additive attention may outperform dot-product attention without scaling, as the dot products can become large, leading to small gradients in the softmax function. To mitigate this, the dot products are scaled by \(\sqrt{d_k}\).

### Multi-Head Attention
Multi-Head Attention enhances the attention mechanism by projecting the queries, keys, and values multiple times (denoted as \(h\) times) using different learned linear projections. Each projection results in dimensions of \(d_k\) for keys and queries, and \(d_v\) for values. The attention function is then applied in parallel across these projected versions, allowing the model to jointly attend to information from different representation subspaces.

This approach results in \(d_v\)-dimensional outputs that have mean 0 and variance 1, improving the model's ability to capture various aspects of the input data. The dot product of the projected queries and keys is computed, which has a mean of 0 and variance of \(d_k\).

Overall, these attention mechanisms are crucial for improving the performance of neural networks, particularly in tasks involving sequence data, such as natural language processing.The text discusses the architecture of a Transformer model, focusing on the multi-head attention mechanism and its applications within the model. Here are the key points:

1. **Multi-Head Attention**: The model employs multi-head attention to allow simultaneous attention to different representation subspaces. The formula for multi-head attention is given as:
\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
\]
where each head is computed as:
\[
\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)
\]
The parameters \(W_i\) are projection matrices, and the model uses \(h = 8\) heads with \(d_k = d_v = d_{model}/h = 64\).

2. **Applications of Attention**: The Transformer uses multi-head attention in three ways:
- **Encoder-Decoder Attention**: Queries come from the decoder, while keys and values come from the encoder, allowing the decoder to attend to all input positions.
- **Self-Attention in Encoder**: All keys, values, and queries come from the encoder's previous layer, enabling each position to attend to all previous positions.
- **Self-Attention in Decoder**: Similar to the encoder, but with a masking mechanism to prevent leftward information flow, preserving the auto-regressive property.

3. **Position-wise Feed-Forward Networks**: Each layer in the encoder and decoder includes a feed-forward network applied independently to each position. This consists of two linear transformations with a ReLU activation:
\[
\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2
\]
The input and output dimensions are \(d_{model} = 512\) and the inner layer has a dimensionality of \(d_f = 2048\).

4. **Embeddings and Softmax**: The model uses learned embeddings to convert input and output tokens into vectors of dimension \(d_{model}\). It shares the weight matrix between the embedding layers and the pre-softmax linear transformation, scaling the weights by \(\sqrt{d_{model}}\).

These components work together to form the basis of the Transformer architecture, enabling effective sequence transduction tasks.**Table 1: Maximum Path Lengths, Per-Layer Complexity, and Minimum Number of Sequential Operations for Different Layer Types**

| Layer Type                     | Complexity per Layer         | Sequential Operations | Maximum Path Length |
|--------------------------------|------------------------------|-----------------------|---------------------|
| Self-Attention                 | O(n² · d)                    | O(1)                  | O(1)                |
| Recurrent                      | O(n · d²)                    | O(n)                  | O(n)                |
| Convolutional                  | O(k · n · d)                 | O(1)                  | O(logₖ(n))         |
| Self-Attention (restricted)    | O(r · n · d)                 | O(1)                  | O(n/r)              |

### 3.5 Positional Encoding
Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension \(d_{model}\) as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed. In this work, we use sine and cosine functions of different frequencies:

\[
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]
\[
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

where \(pos\) is the position and \(i\) is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from \(2\pi\) to \(10000 \cdot 2\pi\). We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset \(k\), \(PE_{pos+k}\) can be represented as a linear function of \(PE_{pos}\).

We also experimented with using learned positional embeddings instead and found that the two versions produced nearly identical results. We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

### 4 Why Self-Attention
In this section, we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations \((x_1, ..., x_n)\) to another sequence of equal length \((z_1, ..., z_n)\), with \(x_i, z_i \in \mathbb{R}^d\), such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention, we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies. Hence, we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires \(O(n)\) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence...The text discusses various aspects of training models for machine translation, particularly focusing on the computational efficiency and interpretability of self-attention mechanisms compared to convolutional layers. It highlights the challenges posed by the dimensionality of input sequences and proposes methods to improve performance, such as restricting self-attention to a local neighborhood. The training data used includes the WMT 2014 English-German and English-French datasets, with specific details on tokenization and batching strategies. The hardware setup for training involves multiple NVIDIA P100 GPUs, and the training schedule is outlined, including the use of the Adam optimizer with a specific learning rate schedule and regularization techniques. The section emphasizes the importance of model interpretability through attention distributions and the distinct tasks learned by individual attention heads.**Table 2: Performance Comparison of the Transformer Model**

The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

| Model                                            | BLEU (EN-DE) | BLEU (EN-FR) | Training Cost (FLOPs) (EN-DE) | Training Cost (FLOPs) (EN-FR) |
|--------------------------------------------------|--------------|--------------|-------------------------------|-------------------------------|
| ByteNet [18]                                     | 23.75        |              |                               |                               |
| Deep-Att + PosUnk [39]                           |              | 39.2         | 1.0 · 10^20                  |                               |
| GNMT + RL [38]                                   | 24.6         | 39.92        | 2.3 · 10^18                  | 1.4 · 10^20                  |
| ConvS2S [9]                                     | 25.16        | 40.46        | 9.6 · 10^19                  | 1.5 · 10^20                  |
| MoE [32]                                        | 26.03        | 40.56        | 2.0 · 10^20                  | 1.2 · 10^20                  |
| Deep-Att + PosUnk Ensemble [39]                  |              | 40.4         |                               | 8.0 · 10^21                  |
| GNMT + RL Ensemble [38]                          | 26.30        | 41.16        | 1.8 · 10^19                  | 1.1 · 10^21                  |
| ConvS2S Ensemble [9]                             | 26.36        | 41.29        | 7.7 · 10^19                  | 1.2 · 10^20                  |
| Transformer (base model)                         | 27.3         | 38.1         | 3.3 · 10^18                  |                               |
| Transformer (big)                                | 28.4         | 41.8         | 2.3 · 10^20                  |                               |

**Residual Dropout**: We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1.

**Label Smoothing**: During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

### 6 Results

#### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.

#### 6.2 Model Variations

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation.Table 3 presents variations on the Transformer architecture, with all metrics derived from the English-to-German translation development set, newstest2013. The perplexities listed are per-wordpiece, based on byte-pair encoding, and should not be compared to per-word perplexities.

| Model | N | dmodel | dff | h | dk | dv | Pdrop | ϵls | train steps | PPL (dev) | BLEU (dev) | params (×10^6) |
|-------|---|--------|-----|---|----|----|-------|-----|-------------|-----------|------------|-----------------|
| base  | 6 | 512    | 2048| 8 | 64 | 64 | 0.1   | 0.1 | 100K        | 4.92      | 25.8       | 65              |
| (A)   | 1 | 512    | 512 |   |    |    |       |     |             | 5.29      | 24.9       |                 |
|       | 4 | 128    | 128 |   |    |    |       |     |             | 5.00      | 25.5       |                 |
|       | 16| 32     | 32  |   |    |    |       |     |             | 4.91      | 25.8       |                 |
|       | 32| 16     | 16  |   |    |    |       |     |             | 5.01      | 25.4       |                 |
| (B)   |   | 16     |     |   |    |    |       |     |             | 5.16      | 25.1       | 58              |
|       |   | 32     |     |   |    |    |       |     |             | 5.01      | 25.4       | 60              |
|       | 2 |        |     |   |    |    |       |     |             | 6.11      | 23.7       | 36              |
|       | 4 |        |     |   |    |    |       |     |             | 5.19      | 25.3       | 50              |
|       | 8 |        |     |   |    |    |       |     |             | 4.88      | 25.5       | 80              |
| (C)   |   | 256    |     | 32| 32 |    |       |     |             | 5.75      | 24.5       | 28              |
|       |   | 1024   |     | 128| 128|   |       |     |             | 4.66      | 26.0       | 168             |
|       |   | 1024   |     |   |    |    |       |     |             | 5.12      | 25.4       | 53              |
|       |   | 4096   |     |   |    |    |       |     |             | 4.75      | 26.2       | 90              |
| (D)   |   |        |     |   |    |    | 0.0   |     |             | 5.77      | 24.6       |                 |
|       |   |        |     |   |    |    | 0.2   |     |             | 4.95      | 25.5       |                 |
|       |   |        |     |   |    |    | 0.0   |     |             | 4.67      | 25.3       |                 |
|       |   |        |     |   |    |    | 0.2   |     |             | 5.47      | 25.7       |                 |
| (E)   |   |        |     |   |    |    | positional embedding instead of sinusoids | | | 4.92 | 25.7 | |
| big   | 6 | 1024   | 4096| 16|    |    | 0.3   |     | 300K        | 4.33      | 26.4       | 213             |

The results indicate that varying the number of attention heads and the dimensions of attention keys and values affects model performance. Specifically, single-head attention results in a 0.9 BLEU score decrease compared to the best configuration, while too many heads also degrade quality. Reducing the attention key size (dk) negatively impacts model quality, suggesting that a more complex compatibility function may be advantageous. Larger models generally perform better, and dropout is effective in mitigating overfitting. The use of learned positional embeddings yields results comparable to the base model.

In the subsequent section, we explore the application of the Transformer model to English constituency parsing, a task characterized by strict structural constraints and longer outputs than inputs. Previous RNN sequence-to-sequence models have struggled to achieve state-of-the-art results in scenarios with limited data. We trained a 4-layer Transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank, comprising approximately 40K training sentences. Additionally, we employed a semi-supervised approach using larger corpora, including high-confidence and BerkleyParser datasets, totaling around 17M sentences. The vocabulary size was set to 16K tokens for the WSJ-only setting and 32K tokens for the semi-supervised setting.

A limited number of experiments were conducted to optimize dropout rates, attention, residual settings, learning rates, and beam sizes on the Section 22 development set, while all other parameters remained consistent with the English-to-German base translation model.The provided text includes a table summarizing the performance of various parsers on English constituency parsing, specifically on Section 23 of the Wall Street Journal (WSJ). The results indicate that the Transformer model, despite not being specifically tuned for the task, performs competitively compared to other models, achieving an F1 score of 91.3 in the WSJ only setting and 92.7 in the semi-supervised setting.

The conclusion highlights the advantages of the Transformer model, which is based entirely on attention mechanisms, allowing for faster training compared to recurrent or convolutional architectures. The authors express enthusiasm for the potential of attention-based models and outline plans for future research, including applying the Transformer to various tasks beyond text and exploring efficient attention mechanisms for handling large inputs and outputs.

The acknowledgments section expresses gratitude to colleagues for their contributions to the work.The text provided appears to be a list of references from a scientific article, specifically focusing on various works related to neural networks, machine translation, and deep learning. Each entry includes the authors, title of the work, publication venue, and year of publication. If you need further assistance or a summary of specific references, please let me know!The text provided appears to be a list of references from a scientific article, specifically related to computational linguistics and natural language processing. Each entry includes the authors, title of the work, publication venue, and other relevant details such as volume, issue, and page numbers.

If you need assistance with a specific aspect of this text, such as summarizing the content, analyzing the references, or extracting particular information, please let me know!It seems that the text you've provided is a mix of OCR output and some formatting artifacts. The content appears to discuss attention mechanisms in a neural network, specifically in the context of an encoder's self-attention layer.

The mention of "long-distance dependencies" suggests that the article is exploring how certain words in a sentence can influence the understanding of other words that are not immediately adjacent, which is a key feature of attention mechanisms in models like Transformers.

If you need further assistance or a specific analysis of this content, please let me know!The text appears to be a fragment from a scientific article discussing attention mechanisms in neural networks, particularly in the context of anaphora resolution. It includes a visual representation (Figure 4) that illustrates the attention heads in a specific layer of the model. The focus is on how these attention heads respond to the word "its," indicating their role in understanding references within the text. The mention of "Input-Input Layer5" suggests a technical discussion related to the architecture of the model being analyzed.

If you need further analysis or a summary of specific sections, please let me know!The text appears to be a fragment from a scientific article discussing the behavior of attention heads in a neural network model, specifically in the context of an encoder self-attention mechanism. The excerpt highlights that different attention heads in layer 5 of the model exhibit distinct behaviors that are related to the structure of sentences. This suggests that the model has learned to perform various tasks through its attention mechanisms.

If you need further analysis or a summary of specific sections, feel free to ask!