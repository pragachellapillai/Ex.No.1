# Ex.No.1 COMPREHENSIVE REPORT ON THE FUNDAMENTALS OF GENERATIVE AI AND LARGE LANGUAGE MODELS (LLMS)

# Aim:
Comprehensive Report on the Fundamentals of Generative AI and Large Language Models (LLMs)

# Explain the foundational concepts of Generative AI.

Generative AI, a subset of artificial intelligence, is a technology that leverages machine learning models to create entirely new data, like text, images, audio, or code, based on patterns learned from existing data sets. It essentially mimics the structure and characteristics of the training data to generate outputs that seem authentic and novel.

# Key Foundational Concepts of Generative AI:

 Generative Models: The core component of generative AI, these 
models are designed to learn the underlying patterns and relationships 
within data, allowing them to generate new data points that adhere to 
those patterns.

 Training Data: A large collection of data used to train the generative 
model. The quality and diversity of the training data significantly 
impacts the quality of generated outputs
.
 Latent Space: An abstract representation of the data in a lower￾dimensional space, where the model learns to encode the essential features of the data. Generating new data involves sampling from this 
latent space.

 Inference Process: Once trained, the model uses the learned patterns to generate new data by sampling from the latent space and decoding it back into the original data format.

# Common Generative AI Techniques:

 Variational Autoencoders (VAEs): Encode input data into a lower￾dimensional latent representation,then decode it back to reconstruct the original data, learning a compressed representation that allows for 
generating new data points.

 Generative Adversarial Networks (GANs): A framework where two neural networks compete against each other – a generator that creates new data, and a discriminator that tries to distinguish between real and 
generated data. This competition pushes both networks to improve, resulting in high-quality generated outputs.

 Transformer-based Models (like GPT-3): A recent advancement in natural language processing, using self-attention mechanisms to learn long-range dependencies in text, which allows for generating coherent 
and contextually relevant text.

# Applications of Generative AI:

 Image Generation: Creating realistic images based on text descriptions 
(e.g., DALL-E, Midjourney)

 Text Generation: Generating creative writing, translating languages, 
writing different styles of text (e.g., ChatGPT)

 Music Generation: Composing new music pieces based on a given style

 Drug Discovery: Generating novel molecular structures for potential drug candidates

 Code Generation: Generating code snippets based on natural language 
prompts (e.g., GitHub Copilot)

# Important Considerations:

 Ethical Concerns: Potential misuse of generative AI for creating deepfakes, misinformation, or biased content.

 Data Privacy: Ensuring the privacy of sensitive information used in training data.

 Model Explainability: Understanding how a generative model makes decisions to mitigate potential biases.

# 2. Focusing on Generative AI architectures. (like transformers).

Large Language Models (LLMs) based on transformers have outperformed the earlier Recurrent Neural Networks (RNNs) in various tasks like sentiment analysis, machine translation, text summarization, etc.

Transformers achieve their unique capabilities from its architecture. This chapter will explain the main ideas of the original transformer model in simple terms to make it easier to understand.

We will focus on the key components that make the transformer: the encoder, the decoder, and the unique attention mechanism that connects them both.

# How do Transformers Work in Generative AI?
Let’s understand how a transformer works −

 First, when we provide a sentence to the transformer, it pays extra attention to the important words in that sentence.

 It then considers all the words simultaneously rather than one after another which helps the transformer to find the dependency between the words in that sentence.

 After that, it finds the relationship between words in that sentence. For example, suppose a sentence is about stars and galaxies then it knows that these words are related.

 Once done, the transformer uses this knowledge to understand the complete story and how words connect with each other.

 With this understanding, the transformer can even predict which word might come next.
# Transformer Architecture in Generative AI:

The transformer has two main components: the encoder and the decoder. Below is a simplified architecture of the transforme

![image](https://github.com/user-attachments/assets/41893b78-beaf-4a4b-a681-bc263e8cf001)

As we can see in the above diagram, there is a residual connection around both the sub-layers, i.e., multi-head attention mechanism and FeedForward Network. The job of these residual connections is to send the unprocessed input x of a sub-layer to a layer normalization function.In this way, the normalized output of each layer can be calculated as below −Layer Normalization (x + Sublayer(x))We will discuss the sulayers, i.e., multi-head attention and FNN, Input embeddings, positional encodings, normalization, and residual connections in detail in subsequent chapters.

# A Layer of the Decoder Stack of the Transformer:

In transformer, the decoder takes the representations generated by the encoder and processes them to generate output sequences. It is just like a translation or a text continuation. Like the encoder, the layers of the decoder of the transformer model are also stacks of layers where each decoder stack layer has the following structure

![image](https://github.com/user-attachments/assets/1b0e6414-9a56-4f13-b75e-4cc321f5fc23)

# Tiny Llama Experiment :
Since the Chinchilla’s laws are not applicable, Tiny Llama experiment tried to compress maximum learning into a small model. The Tiny Llama experiment took a 1 bn parameter model and trained it for 3 trillion tokens (far beyond the 20x recommended by Chinchilla as optimal). However, the model could not 
perform as well compared to either chinchilla or GPT models. This should be obvious as reasoning abilities require multi layer language representations to facilitate emergent abilities and a one billion parameter would not be sufficient to pack all those model learning required.

# Optimal Model :
If our objective is to have good reasoning abilities even if the model size or internal knowledge is just sufficient enough, what should be the model size? Such a model can be trained for optimal dataset size and training tokens to achieve optimal performance. Researchers believe that such a model should 
score at least 60% on MMLU benchmark. It was found that a score of 60% on MMLU implied that the model would exhibit emergent abilities and can perform well as a reasoning engine.Even early GPT-3 and Gopher models could achieve only around 60% MMLU score, but Chinchilla could achieve equivalent MMLU score. Now that chinchilla laws are also not applicable, can we achieve 60% MMLU on much smaller models that can run on a laptop? This was the mission with which Mistral AI was founded, which led to Mistral 8B that had comparable MMLU score with bigger models.


# Result

At present, the evolution of AI Research can be categorized into three 
categories: one driven by business needs that aim to use LLM workflow 
architecture that uses Agents and Tools; the second is based on the possible 
evolution of autonomous agents of the kind of Terminator movie fame - Sky 
net and third is the quest for Artificial General Intelligence (AGI) which can 
supposedly solve problems faced by humanity by helping in research.
While AGI quest seems to go towards bigger and bigger models, business use 
cases might lead to invention of smaller models of the type Llama 8B and 
Mistral 8B which may just serve as reasoning engines. Models that seem to lead 
to autonomous agents might have a size in-between these spectrum.

# Result
