# Natural Language Processing (NLP) Fundamentals

Natural Language Processing (NLP) focuses on enabling computers to understand, analyze, and generate human language. Before modern transformer models such as BERT or GPT became dominant, traditional NLP relied heavily on representing text as numerical vectors.

Machine learning algorithms cannot process raw text directly. Therefore, words, sentences, or entire documents must first be converted into numerical representations.

This section introduces four of the most influential text representation techniques:

1. TF-IDF
2. Word2Vec
3. GloVe
4. FastText

---

# 1. TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF is one of the earliest and most widely used text vectorization techniques.

Rather than learning semantic relationships between words, TF-IDF measures **how important a word is within a document relative to an entire collection of documents (corpus).**

It is commonly used for:

- Search engines
- Document retrieval
- Spam detection
- Text classification
- Keyword extraction

---

## Core Idea

Some words appear frequently in nearly every document.

Examples:

```
the
is
of
and
```

Although these words appear often, they provide very little information.

Conversely, words that occur frequently within one document but rarely across the entire corpus are often highly informative.

TF-IDF assigns larger weights to these informative words.

---

## Term Frequency (TF)

Term Frequency measures how often a word appears inside a document.

To compute TF:

1. Count how many times the word appears.
2. Divide by the total number of words in the document.

Example:

```
Document

Machine learning is fun.
Machine learning is useful.
```

The word

```
Machine
```

appears twice.

Words appearing more frequently receive larger TF values.

---

## Inverse Document Frequency (IDF)

Some words appear in almost every document.

These words contribute little toward distinguishing documents.

IDF reduces their importance.

To compute IDF conceptually:

1. Count the total number of documents.
2. Count how many documents contain the word.
3. Words appearing in many documents receive lower scores.
4. Rare words receive higher scores.

---

## Combining TF and IDF

The final importance score is obtained by multiplying:

- Term Frequency
- Inverse Document Frequency

Consequently,

a word receives a high TF-IDF score if:

- it appears frequently inside one document
- but appears rarely throughout the corpus.

---

## Bag-of-Words Representation

TF-IDF is built upon the Bag-of-Words model.

Each document becomes a vector whose length equals the vocabulary size.

Example:

Vocabulary

```
Machine
Learning
Python
Coffee
```

Document:

```
Machine Learning Python
```

Vector:

```
Machine    1
Learning   1
Python     1
Coffee     0
```

TF-IDF replaces these binary or frequency counts with weighted importance scores.

---

## Advantages

- Simple
- Fast
- Easy to interpret
- Works well for document classification
- No training required

---

## Limitations

- Ignores word order
- Ignores context
- Cannot capture semantic similarity
- Produces sparse vectors
- Vocabulary grows rapidly

---

# 2. Word2Vec

Word2Vec is a neural network-based word embedding algorithm introduced by Google in 2013.

Unlike TF-IDF,

Word2Vec learns **semantic relationships** between words.

Instead of representing every word using a sparse vocabulary vector,

each word becomes a dense numerical vector.

Words with similar meanings receive similar vectors.

---

## Core Idea

Consider the words

```
King

Queen

Prince

Princess
```

These words frequently appear in similar contexts.

Word2Vec learns that their vectors should be close together.

Likewise,

```
Dog

Cat
```

become similar,

while

```
Dog

Car
```

remain distant.

---

## Distributional Hypothesis

Word2Vec is based on a famous linguistic principle:

> Words appearing in similar contexts tend to have similar meanings.

For example,

```
The cat drinks milk.

The dog drinks milk.
```

The words

```
cat

dog
```

appear in nearly identical contexts.

Therefore,

their vectors become similar.

---

# CBOW (Continuous Bag of Words)

CBOW predicts the current word using surrounding context words.

Example:

```
The ___ drinks milk.
```

Input:

```
The

drinks

milk
```

Prediction:

```
cat
```

Training repeats this process for millions of sentences.

---

# Skip-Gram

Skip-Gram performs the opposite task.

Given the center word,

predict the surrounding context.

Example:

Input

```
cat
```

Predict

```
The

drinks

milk
```

Skip-Gram generally performs better on rare words,

while CBOW is usually faster.

---

## Neural Network Structure

Word2Vec trains a very small neural network.

It contains:

- Input layer
- Hidden embedding layer
- Output layer

After training,

the hidden layer weights become the word embeddings.

The network itself is discarded.

Only the learned vectors are retained.

---

## Vector Similarity

Similarity between words is typically measured using **Cosine Similarity**.

Instead of comparing magnitudes,

Cosine Similarity compares vector directions.

If two vectors point in nearly the same direction,

their corresponding words have similar meanings.

---

## Famous Example

One remarkable property of Word2Vec is that relationships become encoded mathematically.

For example,

```
King

minus

Man

plus

Woman
```

produces a vector very close to

```
Queen
```

This demonstrates that embeddings capture semantic relationships rather than simple word frequencies.

---

## Advantages

- Learns semantic similarity
- Dense vector representations
- Small memory footprint
- Fast inference
- Useful for many downstream NLP tasks

---

## Limitations

- One vector per word
- Cannot distinguish multiple meanings

Example:

```
bank
```

always has one embedding,

regardless of whether it refers to

- a financial institution
- a river bank.

---

# 3. GloVe (Global Vectors for Word Representation)

GloVe was developed by Stanford University.

Like Word2Vec,

GloVe produces dense word embeddings.

However,

its training strategy differs fundamentally.

---

## Core Idea

Word2Vec learns by predicting nearby words.

GloVe instead learns from **global word co-occurrence statistics.**

Rather than examining one sentence at a time,

it analyzes the entire corpus simultaneously.

---

## Co-occurrence Matrix

GloVe begins by constructing a large matrix.

Each entry records how frequently two words appear together.

Example:

```
           king queen man woman

king         -

queen       high

man         high

woman      medium
```

Words frequently appearing together receive larger counts.

---

## Training

The model attempts to learn embeddings whose relationships explain the observed co-occurrence frequencies.

Rather than predicting neighboring words,

it minimizes the difference between:

- observed word co-occurrences
- predicted co-occurrences

---

## Advantages

- Captures global statistics
- Excellent semantic embeddings
- Stable training
- Often outperformed Word2Vec on classical benchmarks

---

## Limitations

- Requires constructing a large co-occurrence matrix
- Memory intensive for massive vocabularies
- Produces only one embedding per word

---

# 4. FastText

FastText was developed by Facebook AI Research.

It extends Word2Vec by representing words using **character-level subwords** rather than treating every word as a completely independent token.

This dramatically improves handling of:

- Rare words
- Misspellings
- Morphologically rich languages

---

## Core Idea

Instead of learning one vector per word,

FastText learns vectors for small character sequences called **n-grams**.

Example:

Word:

```
learning
```

Character n-grams might include:

```
lea

ear

arn

rni

nin

ing
```

The final word embedding is obtained by combining:

- the word vector
- all character n-gram vectors

---

## Why Subwords Matter

Suppose the training corpus never contains

```
unhappiness
```

Word2Vec cannot create an embedding.

FastText can.

It already understands:

```
un

happy

ness
```

Therefore,

it constructs a reasonable embedding even for unseen words.

---

## Morphological Languages

FastText performs especially well for languages with rich morphology.

Examples include:

- Turkish
- Finnish
- Hungarian
- German

Many words share common prefixes and suffixes.

Subword modeling captures these relationships naturally.

---

## Comparison with Word2Vec

Suppose the model has learned

```
run

running

runner

runs
```

Word2Vec learns four unrelated vectors.

FastText recognizes that these words share many character sequences.

Their embeddings therefore become naturally related.

---

## Advantages

- Handles unseen words
- Better rare-word representations
- Excellent for morphologically rich languages
- Robust against spelling variations
- Dense semantic embeddings

---

## Limitations

- Larger models
- Slightly slower training
- More memory usage than Word2Vec

---

# Comparison

| Method | Learns Meaning | Uses Context | Handles Rare Words | Sparse/Dense | Training Required |
|---------|---------------|--------------|--------------------|--------------|------------------|
| TF-IDF | No | No | No | Sparse | No |
| Word2Vec | Yes | Yes | Limited | Dense | Yes |
| GloVe | Yes | Yes | Limited | Dense | Yes |
| FastText | Yes | Yes | Yes | Dense | Yes |

---

# Choosing the Right Method

### Use TF-IDF when:

- Building search engines
- Document classification
- Keyword extraction
- Working with small datasets
- Model interpretability is important

---

### Use Word2Vec when:

- Semantic similarity matters
- Word embeddings are needed
- Building recommendation systems
- Classical NLP pipelines

---

### Use GloVe when:

- Global corpus statistics are important
- Pretrained embeddings are desired
- Working with traditional NLP tasks

---

### Use FastText when:

- Working with rare words
- Handling spelling mistakes
- Processing morphologically rich languages such as Turkish
- Out-of-vocabulary words are common

---

# Summary

Traditional NLP methods represent text numerically before applying machine learning algorithms.

- **TF-IDF** measures word importance using frequency statistics but ignores semantic meaning.
- **Word2Vec** learns dense embeddings by predicting neighboring words, capturing semantic similarity.
- **GloVe** learns embeddings from global word co-occurrence patterns across an entire corpus.
- **FastText** extends Word2Vec by learning character-level subword representations, making it robust to rare and unseen words.

Although transformer-based models such as BERT and GPT have largely replaced these techniques in state-of-the-art NLP systems, TF-IDF, Word2Vec, GloVe, and FastText remain fundamental concepts for understanding modern language representation and continue to be widely used in many practical applications.