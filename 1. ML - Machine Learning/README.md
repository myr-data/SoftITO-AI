# Machine Learning Fundamentals

---

# 1. Exploratory Data Analysis (EDA) & Data Wrangling

Before training any machine learning model, it is essential to understand and prepare the dataset. Even the most advanced algorithms cannot produce reliable results if the underlying data is incomplete, inconsistent, or noisy.

Data preprocessing generally consists of two complementary stages:

- **Exploratory Data Analysis (EDA):** Understanding the dataset.
- **Data Wrangling:** Cleaning and transforming the dataset.

In industry, these stages often consume more time than the model-building process itself.

---

## What is Exploratory Data Analysis (EDA)?

Exploratory Data Analysis is the process of examining a dataset before applying any machine learning algorithms.

The primary goal is to answer questions such as:

- How many samples are available?
- How many features exist?
- What data types are present?
- Are there missing values?
- Are there duplicated records?
- Are there outliers?
- Are some features highly correlated?
- Is the target variable balanced?

EDA helps identify problems before they negatively affect model performance.

---

## Descriptive Statistics

EDA begins by calculating descriptive statistics.

### Mean

The mean is the average value of a feature.

To calculate it:

1. Add together every value in the column.
2. Divide the total by the number of observations.

Example:

```
Scores:

70
80
90

Mean

(70 + 80 + 90) / 3 = 80
```

The mean represents the center of the distribution.

---

### Median

The median is the middle value after sorting the observations.

Example:

```
10 15 20 25 500
```

Median:

```
20
```

Unlike the mean, the median is less affected by extremely large or small values.

---

### Variance

Variance measures how spread out the data is.

To calculate variance:

1. Compute the mean.
2. Measure how far every value is from the mean.
3. Square every difference.
4. Compute the average of those squared differences.

A larger variance indicates that observations are more dispersed.

---

### Standard Deviation

Standard deviation is simply the square root of variance.

Unlike variance, it is measured using the original unit of the data, making it much easier to interpret.

---

### Correlation

Correlation measures how strongly two variables change together.

Examples:

```
Temperature ↑
Ice Cream Sales ↑
```

Positive correlation.

```
Outside Temperature ↑
Heating Bill ↓
```

Negative correlation.

Correlation values typically range between:

```
+1  Perfect positive relationship

 0  No linear relationship

-1  Perfect negative relationship
```

Correlation matrices are commonly visualized using heatmaps.

---

## Data Visualization

Visualization often reveals patterns that statistics alone cannot.

Common visualization techniques include:

- Histograms
- Box plots
- Scatter plots
- Pair plots
- Correlation heatmaps
- Count plots

These help identify:

- skewed distributions
- clusters
- anomalies
- relationships between variables

---

## Data Wrangling

Data wrangling transforms raw data into a clean dataset suitable for machine learning.

Typical tasks include:

### Handling Missing Values

Missing values can significantly reduce model quality.

Common strategies include:

- Remove rows containing missing values
- Remove columns with excessive missing data
- Replace missing values using the column mean
- Replace using the median
- Replace using the most common value (mode)
- Predict missing values using another model

The appropriate strategy depends on the dataset.

---

### Removing Duplicates

Duplicate records bias statistical analysis and machine learning models.

Duplicates should usually be removed before training.

---

### Handling Outliers

Outliers are observations that differ significantly from the rest of the data.

Example:

```
20
22
23
24
26
450
```

The value 450 is likely an outlier.

Outliers may represent:

- measurement errors
- data entry mistakes
- genuinely rare events

Whether they should be removed depends on the application.

---

### Encoding Categorical Variables

Machine learning algorithms require numerical input.

Suppose we have

```
Color

Red
Blue
Green
```

One possible encoding is

```
Red = 0
Blue = 1
Green = 2
```

However, this introduces an artificial ordering.

Instead, One-Hot Encoding creates separate binary columns.

```
Red Blue Green

1    0    0

0    1    0

0    0    1
```

This avoids implying numerical relationships.

---

### Feature Scaling

Some algorithms depend heavily on numerical distances.

Suppose one feature ranges from

```
0–1
```

while another ranges from

```
0–1,000,000
```

The second feature dominates distance calculations.

Scaling prevents this problem.

Two common methods are:

### Min-Max Scaling

This transforms values so they lie between 0 and 1.

The process is:

1. Find the minimum value.
2. Find the maximum value.
3. Measure how far each observation is from the minimum.
4. Divide by the total range.

---

### Standardization

Standardization transforms features so that:

- average becomes zero
- standard deviation becomes one

This is especially useful for:

- Logistic Regression
- SVM
- Neural Networks
- PCA

---

## Why EDA Matters

EDA identifies problems before training begins.

Without proper exploration, models may suffer from:

- poor accuracy
- overfitting
- biased predictions
- unstable behavior

Understanding the dataset is often more valuable than choosing a more sophisticated algorithm.

---

# 2. Linear Regression

Linear Regression is one of the simplest supervised learning algorithms.

It predicts a continuous numerical value.

Typical applications include:

- House price prediction
- Salary prediction
- Sales forecasting
- Temperature prediction

---

## Core Idea

Linear Regression assumes that the relationship between the input features and the target variable can be approximated using a straight line (or hyperplane in higher dimensions).

Suppose we wish to predict house prices.

```
House Size

100 m²

Price

$150,000
```

Larger houses generally cost more.

Linear Regression attempts to find the line that best describes this relationship.

---

## Mathematical Intuition

The model makes predictions using two components:

- an intercept (starting value)
- one weight for every feature

Prediction proceeds as follows:

1. Multiply every feature by its corresponding weight.
2. Add all of these values together.
3. Add the intercept.
4. The result becomes the prediction.

Each weight represents how strongly that feature influences the prediction.

---

## Cost Function

The model must determine the best weights.

To evaluate a set of weights:

1. Predict every training sample.
2. Compare predictions with actual values.
3. Compute the prediction errors.
4. Square every error.
5. Average all squared errors.

This quantity is called the Mean Squared Error (MSE).

Large prediction errors contribute much more than small errors because errors are squared.

---

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize prediction error.

Imagine standing on a mountain covered in fog.

Your goal is to reach the lowest point.

You cannot see the entire mountain.

Instead:

1. Measure the slope beneath your feet.
2. Move slightly downhill.
3. Repeat until no further improvement is possible.

Machine learning performs the same process numerically.

During every iteration:

- compute prediction error
- determine which direction reduces error
- slightly update every weight
- repeat until convergence

---

## Assumptions

Linear Regression assumes:

- approximately linear relationships
- independent observations
- constant variance
- minimal multicollinearity
- normally distributed residual errors

---

## Advantages

- Very simple
- Fast training
- Highly interpretable
- Good baseline model

---

## Limitations

- Cannot model nonlinear relationships
- Sensitive to outliers
- Assumes linearity
- Poor performance on complex datasets

---

# 3. Logistic Regression

Despite its name, Logistic Regression performs **classification**, not regression.

Instead of predicting a continuous number, it predicts probabilities.

Applications include:

- Spam detection
- Fraud detection
- Disease diagnosis
- Customer churn prediction

---

## Why Linear Regression Cannot Be Used

Suppose Linear Regression predicts

```
1.8

-0.6

3.2
```

These are impossible probabilities.

Probabilities must lie between

```
0 and 1
```

Logistic Regression solves this problem using the Sigmoid function.

---

## Sigmoid Function

The model first computes a weighted sum of the input features, just as Linear Regression does.

Instead of returning that value directly, it passes the value through the Sigmoid function.

The Sigmoid function smoothly compresses any input into a value between 0 and 1.

Very negative values become numbers close to 0.

Very positive values become numbers close to 1.

Values near zero become approximately 0.5.

---

## Decision Boundary

After computing a probability,

a threshold determines the class.

Most commonly:

```
Probability ≥ 0.5

Positive Class

Probability < 0.5

Negative Class
```

This threshold can be adjusted depending on the application.

---

## Binary Cross Entropy Loss

Instead of minimizing squared error,

Logistic Regression minimizes Binary Cross Entropy.

The idea is simple:

- Correct predictions receive a small penalty.
- Confident incorrect predictions receive a very large penalty.

This encourages accurate probability estimates.

---

## Gradient Descent

Training follows the same general idea as Linear Regression.

During every iteration:

- compute predicted probabilities
- compare with actual labels
- measure prediction error
- update model weights
- repeat

---

## Advantages

- Fast
- Simple
- Produces probabilities
- Easy to interpret

---

## Limitations

- Linear decision boundary
- Cannot model highly complex relationships
- Sensitive to correlated features

---

# 4. Decision Tree & Random Forest

Decision Trees classify data by repeatedly asking questions.

Example:

```
Age > 30?

Yes

Income > 50k?

No

Reject Loan
```

Every decision divides the dataset into increasingly pure subsets.

---

## Tree Structure

A Decision Tree consists of:

- Root node
- Internal nodes
- Branches
- Leaf nodes

Leaves contain the final prediction.

---

## Choosing the Best Split

The algorithm evaluates many possible questions.

For every candidate split,

it measures how much uncertainty decreases.

Common impurity measures include:

- Entropy
- Information Gain
- Gini Impurity

The split producing the purest child nodes is selected.

---

## Entropy

Entropy measures disorder.

If every observation belongs to the same class,

entropy is low.

If classes are evenly mixed,

entropy is high.

Decision Trees attempt to reduce entropy after every split.

---

## Information Gain

Information Gain measures how much uncertainty decreases after splitting.

The algorithm:

1. Computes uncertainty before splitting.
2. Computes uncertainty after splitting.
3. Selects the split producing the largest reduction.

---

## Gini Impurity

Many implementations use Gini Impurity instead.

Gini estimates how likely a random observation would be classified incorrectly.

Lower Gini values indicate purer nodes.

---

# Random Forest

A Random Forest combines many Decision Trees.

Instead of training one tree,

it trains hundreds of trees.

Each tree receives:

- a random subset of observations
- a random subset of features

Each tree produces its own prediction.

The final result is:

Classification:

Majority vote.

Regression:

Average prediction.

---

## Why Random Forest Works Better

Individual Decision Trees easily overfit.

Random Forest reduces this by averaging many independent trees.

Individual mistakes tend to cancel out.

---

## Advantages

- High accuracy
- Robust against overfitting
- Handles nonlinear relationships
- Requires little preprocessing

---

## Limitations

- Less interpretable
- Larger memory usage
- Slower predictions

---

# 5. K-Nearest Neighbors (KNN)

KNN is one of the simplest machine learning algorithms.

Unlike most models,

it performs almost no training.

Instead,

it stores the training dataset.

When a new observation arrives,

it searches for similar examples.

---

## Distance

To determine similarity,

KNN computes distances.

Most commonly,

Euclidean distance.

Imagine plotting points on graph paper.

The closest points are considered the most similar.

---

## Algorithm

For every new observation:

1. Compute distances to every training sample.
2. Sort by distance.
3. Select the K nearest neighbors.
4. Classification:
   Majority vote.
5. Regression:
   Average target value.

---

## Choosing K

Small K:

- very flexible
- sensitive to noise

Large K:

- smoother predictions
- higher bias

---

## Curse of Dimensionality

As the number of features increases,

all observations become similarly distant.

Distance calculations become less meaningful.

Consequently,

KNN performs poorly on very high-dimensional datasets.

---

## Advantages

- Extremely simple
- No training phase
- Naturally handles nonlinear boundaries

---

## Limitations

- Slow predictions
- Large memory usage
- Sensitive to feature scaling
- Suffers in high dimensions

---

# 6. Support Vector Machine (SVM)

Support Vector Machines attempt to separate classes using the widest possible margin.

Rather than simply finding any separating line,

SVM searches for the one that leaves the largest distance between classes.

---

## Hyperplane

In two dimensions,

the decision boundary is a line.

In three dimensions,

it becomes a plane.

In higher dimensions,

it is called a hyperplane.

---

## Margin

Suppose two classes can be separated by many different lines.

SVM selects the line that leaves the greatest empty space between them.

A larger margin generally improves generalization.

---

## Support Vectors

Only a few training samples determine the final decision boundary.

These critical observations are called support vectors.

Removing other observations often has little effect.

---

## Soft Margin

Real-world datasets are rarely perfectly separable.

Instead of requiring every point to be correctly classified,

SVM allows some violations.

A parameter called C controls the trade-off.

Small C:

- wider margin
- more classification errors allowed

Large C:

- narrower margin
- fewer training errors

---

## Kernel Trick

Some datasets cannot be separated using a straight line.

Instead of manually creating new features,

SVM uses kernel functions.

A kernel implicitly transforms the data into a higher-dimensional space where separation becomes possible.

Common kernels include:

- Linear
- Polynomial
- Radial Basis Function (RBF)
- Sigmoid

---

## Advantages

- Excellent for high-dimensional data
- Effective with limited training data
- Robust against overfitting

---

## Limitations

- Slow on very large datasets
- Sensitive to parameter tuning
- Less interpretable

---

# 7. AdaBoost, XGBoost & LightGBM

Boosting is an ensemble learning technique where models are trained **sequentially**.

Each new model attempts to improve the mistakes made by previous models.

Instead of relying on one powerful model,

boosting combines many relatively simple models into a highly accurate predictor.

Most boosting algorithms use shallow Decision Trees as their base learners.

---

# AdaBoost (Adaptive Boosting)

AdaBoost was one of the first successful boosting algorithms.

Its main idea is simple:

After each Decision Tree is trained,

the algorithm identifies which training samples were classified incorrectly.

Those difficult samples receive higher importance during the next iteration.

As training continues,

later trees focus increasingly on difficult observations.

---

## Training Process

1. Every sample initially receives equal importance.
2. Train a small Decision Tree.
3. Measure which samples were classified incorrectly.
4. Increase the importance of those samples.
5. Train another tree using the updated sample weights.
6. Repeat many times.
7. Combine all trees into a weighted vote.

Trees with better performance contribute more strongly to the final prediction.

---

## Advantages

- Easy to understand
- Good accuracy on structured datasets
- Naturally focuses on difficult examples

---

## Limitations

- Sensitive to noisy data
- Sensitive to incorrect labels
- Sequential training makes parallelization difficult

---

# Gradient Boosting

AdaBoost modifies the importance of training samples.

Gradient Boosting takes a different approach.

Instead of emphasizing difficult samples,

each new Decision Tree learns to predict the **remaining prediction errors** (called residuals) of the previous ensemble.

Each iteration gradually reduces the remaining error.

This idea forms the foundation of XGBoost, LightGBM and CatBoost.

---

# XGBoost (Extreme Gradient Boosting)

XGBoost is one of the most successful machine learning algorithms for structured data.

It improves ordinary Gradient Boosting by introducing:

- Regularization
- Faster optimization
- Parallel computation
- Automatic handling of missing values
- Efficient tree pruning

---

## How It Works

Instead of growing trees independently,

XGBoost builds them sequentially.

Each new tree attempts to correct the prediction errors made by all previous trees.

Unlike standard Gradient Boosting,

XGBoost not only considers how large the prediction error is,

but also how quickly the error is changing.

Using this additional information allows optimization to converge faster and often produces more accurate models.

---

## Regularization

Complex trees often memorize training data.

To reduce overfitting,

XGBoost penalizes:

- excessively deep trees
- excessive numbers of leaves
- large leaf weights

This encourages simpler models that generalize better.

---

## Advantages

- Extremely accurate
- Robust against overfitting
- Handles missing values
- Strong performance on tabular datasets

---

## Limitations

- Many hyperparameters
- Higher memory usage
- Slower than LightGBM

---

# LightGBM (Light Gradient Boosting Machine)

LightGBM is Microsoft's implementation of Gradient Boosting.

Its primary objective is speed.

Instead of evaluating every possible split,

LightGBM groups continuous values into discrete bins.

This dramatically reduces computation.

---

## Histogram-Based Learning

Rather than considering every unique numerical value,

LightGBM first places values into intervals.

Instead of examining thousands of candidate split points,

the algorithm only evaluates a relatively small number of histogram bins.

This greatly accelerates training while consuming less memory.

---

## Leaf-wise Growth

Most Decision Tree algorithms expand trees level by level.

LightGBM grows whichever leaf currently reduces prediction error the most.

This often produces deeper but more accurate trees.

---

## Advantages

- Extremely fast
- Low memory usage
- Excellent for massive datasets
- High predictive accuracy

---

## Limitations

- Can overfit small datasets
- More sensitive to hyperparameter tuning
- Deep trees may require careful regularization

---

# Comparison

| Algorithm | Main Idea | Strength | Weakness |
|------------|-----------|----------|-----------|
| AdaBoost | Increase importance of misclassified samples | Simple and effective | Sensitive to noise |
| Gradient Boosting | Learn residual errors | Strong predictive power | Slower training |
| XGBoost | Optimized Gradient Boosting with regularization | Excellent accuracy | More complex |
| LightGBM | Histogram-based Gradient Boosting | Extremely fast | Can overfit small datasets |

---

## Summary

Boosting algorithms improve model performance by training Decision Trees sequentially.

Their progression can be summarized as follows:

- **AdaBoost** improves future trees by increasing the importance of misclassified samples.
- **Gradient Boosting** improves future trees by predicting the remaining residual errors.
- **XGBoost** enhances Gradient Boosting with regularization, second-order optimization, and efficient tree construction.
- **LightGBM** further optimizes Gradient Boosting using histogram-based learning and leaf-wise tree growth to achieve exceptional speed on large datasets.

# Unsupervised Learning & Dimensionality Reduction

Unlike supervised learning, **unsupervised learning** algorithms are trained without labeled data. Instead of learning a mapping between inputs and known outputs, they attempt to discover hidden patterns, structures, or relationships within the data.

Common objectives include:

- Clustering similar observations
- Detecting anomalies
- Discovering latent structures
- Reducing dimensionality
- Visualizing high-dimensional datasets

This section introduces six of the most widely used unsupervised learning techniques.

---

# 1. K-Means Clustering

K-Means is one of the simplest and most widely used clustering algorithms.

Its objective is to divide a dataset into **K distinct groups (clusters)** such that observations inside the same cluster are as similar as possible, while observations in different clusters are as different as possible.

Unlike classification algorithms, K-Means does not know the correct answers beforehand.

Typical applications include:

- Customer segmentation
- Market analysis
- Document clustering
- Image compression
- Recommendation systems
- Data preprocessing

---

## Core Idea

Imagine hundreds of customer locations on a map.

Without knowing anything about them, K-Means attempts to group nearby customers into several natural regions.

Each region is represented by a point called a **centroid**, which is simply the average location of all observations assigned to that cluster.

---

## Training Process

K-Means follows an iterative optimization procedure:

### Step 1

Choose the desired number of clusters, **K**.

For example,

```
K = 3
```

means the algorithm should divide the dataset into three clusters.

---

### Step 2

Randomly initialize K centroids.

Initially, these centroid locations are simply guesses.

---

### Step 3

Assign every observation to its nearest centroid.

Similarity is usually measured using Euclidean distance.

Every observation joins the closest cluster.

---

### Step 4

Recalculate each centroid.

For every cluster:

1. Compute the average of all feature values belonging to that cluster.
2. Move the centroid to this new average position.

---

### Step 5

Repeat Steps 3 and 4 until the centroids stop changing significantly.

This point is called **convergence**.

---

## Distance Metric

K-Means typically measures similarity using Euclidean distance.

Conceptually:

1. Compare two observations feature by feature.
2. Measure the difference for each feature.
3. Square every difference.
4. Sum all squared differences.
5. Take the square root.

Smaller distances indicate more similar observations.

---

## Choosing K

One limitation of K-Means is that the number of clusters must be chosen beforehand.

Common methods include:

### Elbow Method

Train K-Means for several values of K.

For each value:

- compute the total within-cluster error.

As K increases, the error decreases.

Choose the point where further increases produce only small improvements.

This resembles the "elbow" of a curve.

---

### Silhouette Score

Measures how well each observation fits its assigned cluster compared to neighboring clusters.

Higher scores indicate better-defined clusters.

---

## Computational Complexity

For each iteration,

the algorithm compares every sample with every centroid.

Training complexity is approximately proportional to:

- Number of samples
- Number of features
- Number of clusters
- Number of iterations

This makes K-Means efficient even for relatively large datasets.

---

## Advantages

- Simple
- Fast
- Easy to interpret
- Scales well
- Excellent baseline clustering algorithm

---

## Limitations

- Requires choosing K beforehand
- Sensitive to initialization
- Sensitive to outliers
- Assumes approximately spherical clusters
- Performs poorly when clusters have very different sizes or densities

---

# 2. Hierarchical Clustering (Agglomerative)

Hierarchical Clustering constructs a hierarchy of nested clusters rather than producing a single partition.

The result is visualized using a **dendrogram**, allowing users to decide later how many clusters should exist.

Agglomerative clustering is the most common hierarchical approach.

It follows a **bottom-up** strategy.

---

## Core Idea

Initially,

every observation is treated as its own cluster.

The algorithm repeatedly merges the two closest clusters until only one cluster remains.

The entire merging process is recorded in the dendrogram.

---

## Training Process

### Step 1

Treat every sample as an independent cluster.

If there are

```
100 samples
```

the algorithm begins with

```
100 clusters.
```

---

### Step 2

Compute distances between all clusters.

---

### Step 3

Merge the two closest clusters.

---

### Step 4

Recompute distances.

---

### Step 5

Repeat until only one cluster remains.

---

## Linkage Methods

When clusters contain multiple observations,

the distance between clusters is no longer obvious.

Different linkage methods define this distance differently.

---

### Single Linkage

Uses the distance between the two closest observations.

This often creates long chain-like clusters.

---

### Complete Linkage

Uses the distance between the two farthest observations.

This generally produces compact clusters.

---

### Average Linkage

Uses the average distance between all observations in both clusters.

Provides a compromise between Single and Complete Linkage.

---

### Ward Linkage

Rather than minimizing distance,

Ward linkage merges clusters that produce the smallest increase in overall variance.

This often produces balanced, compact clusters.

---

## Dendrogram

A dendrogram is a tree showing the entire merging process.

Cutting the dendrogram at different heights produces different numbers of clusters.

Unlike K-Means,

the number of clusters does not need to be specified before training.

---

## Advantages

- No need to specify K initially
- Produces interpretable dendrograms
- Supports many distance metrics

---

## Limitations

- Computationally expensive
- Poor scalability for very large datasets
- Sensitive to noisy observations
- Early merge decisions cannot be reversed

---

# 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN groups observations according to **density** rather than distance to centroids.

Instead of assuming clusters are spherical,

DBSCAN discovers regions containing many nearby observations.

It is also capable of identifying **outliers automatically**.

---

## Core Idea

Imagine a city map.

Dense neighborhoods naturally form clusters.

Isolated houses remain outside every cluster.

DBSCAN follows exactly this principle.

---

## Key Parameters

### Epsilon (ε)

Defines the maximum neighborhood radius.

Only observations within this radius are considered neighbors.

---

### Minimum Points (MinPts)

Defines the minimum number of neighboring observations required to form a dense region.

---

## Point Types

Every observation belongs to one of three categories.

### Core Point

Has enough neighboring observations to form a dense region.

---

### Border Point

Does not have enough neighbors itself,

but lies within the neighborhood of a Core Point.

---

### Noise Point

Does not belong to any cluster.

These observations are treated as outliers.

---

## Training Process

1. Choose an unvisited observation.
2. Find all neighbors within ε.
3. If enough neighbors exist,
   create a new cluster.
4. Expand the cluster by visiting neighboring Core Points.
5. Continue until every observation has been processed.

---

## Advantages

- Automatically detects outliers
- No need to specify K
- Discovers arbitrary cluster shapes
- Works well for spatial datasets

---

## Limitations

- Sensitive to ε selection
- Sensitive to MinPts selection
- Performs poorly in very high-dimensional spaces
- Struggles when cluster densities vary significantly

---

# 4. Gaussian Mixture Models (GMM)

Unlike K-Means,

Gaussian Mixture Models perform **probabilistic clustering**.

Instead of assigning every observation to exactly one cluster,

GMM estimates the probability that an observation belongs to each cluster.

---

## Core Idea

Suppose a customer lies between two market segments.

K-Means forces the customer into exactly one cluster.

GMM instead produces probabilities such as:

```
Cluster A

15%

Cluster B

80%

Cluster C

5%
```

This is called **soft clustering**.

---

## Gaussian Distribution

Each cluster is modeled as a Gaussian (Normal) distribution.

Every Gaussian is described by:

- Mean (center)
- Variance (spread)
- Covariance (shape and orientation)

Unlike K-Means,

clusters may become elliptical rather than perfectly circular.

---

## Expectation-Maximization (EM)

GMM parameters are estimated using the Expectation-Maximization algorithm.

The algorithm repeatedly alternates between two stages.

### Expectation Step

Estimate the probability that every observation belongs to every cluster.

---

### Maximization Step

Update:

- Cluster centers
- Covariance matrices
- Cluster probabilities

using the current assignments.

Repeat until convergence.

---

## Advantages

- Soft probabilistic assignments
- Models elliptical clusters
- More flexible than K-Means

---

## Limitations

- Slower training
- Sensitive to initialization
- Assumes Gaussian-shaped clusters

---

# 5. Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is one of the most widely used **dimensionality reduction** techniques.

Instead of clustering observations,

PCA transforms the original features into a smaller set of new features called **Principal Components**.

The objective is to preserve as much information as possible while reducing dimensionality.

---

## Why Reduce Dimensions?

High-dimensional datasets often contain:

- Redundant features
- Correlated variables
- Noise
- Expensive computations

Reducing dimensionality often improves:

- Training speed
- Memory usage
- Visualization
- Generalization

---

## Core Idea

Suppose a dataset contains

```
100 features.
```

Many of these may carry nearly identical information.

PCA identifies new directions through the data that capture the greatest variation.

Instead of keeping all 100 original features,

the dataset may be represented using only

```
10 principal components
```

while preserving most of the useful information.

---

## Training Process

### Step 1

Center the dataset.

Subtract the average value from every feature.

This ensures every feature has an average of zero.

---

### Step 2

Measure how features vary together.

Features that change together contain redundant information.

---

### Step 3

Find the directions that explain the greatest variation.

These become the principal components.

---

### Step 4

Sort components from most informative to least informative.

---

### Step 5

Keep only the first few components.

The remaining components are discarded.

---

## Explained Variance

Every principal component explains some percentage of the total dataset variation.

For example:

```
Component 1

52%

Component 2

24%

Component 3

11%
```

The first two components together preserve

```
76%
```

of the original information.

---

## Advantages

- Reduces dimensionality
- Removes redundant information
- Speeds up training
- Reduces multicollinearity
- Useful preprocessing step

---

## Limitations

- Linear technique
- Reduced interpretability
- Sensitive to feature scaling
- Information is inevitably lost

---

# 6. t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction algorithm designed primarily for **visualization**.

Unlike PCA,

its goal is not compression,

but producing meaningful two-dimensional or three-dimensional visualizations.

It is particularly popular for:

- Word embeddings
- Image embeddings
- Neural network feature visualization
- Cluster exploration

---

## Core Idea

Suppose every observation has

```
768 features
```

such as BERT embeddings.

Humans cannot visualize 768-dimensional space.

t-SNE projects the data into

```
2D

or

3D
```

while attempting to preserve local neighborhoods.

Observations that were close together originally remain close together after projection.

---

## Training Process

### Step 1

Measure similarities between observations in the original high-dimensional space.

Nearby observations receive high similarity scores.

---

### Step 2

Create random positions in a low-dimensional space.

Initially,

the layout is essentially random.

---

### Step 3

Adjust the positions repeatedly.

Observations that were close together originally are pulled closer.

Observations that were far apart are pushed apart.

---

### Step 4

Continue updating until the low-dimensional layout closely reflects the original neighborhood relationships.

---

## Difference Between PCA and t-SNE

| PCA | t-SNE |
|------|--------|
| Linear | Nonlinear |
| Fast | Slower |
| Preserves global variance | Preserves local neighborhoods |
| Suitable for preprocessing | Suitable for visualization |
| Deterministic | Random initialization |

---

## Advantages

- Excellent visualization quality
- Reveals hidden clusters
- Preserves local relationships
- Very useful for embedding visualization

---

## Limitations

- Computationally expensive
- Slow on large datasets
- Not suitable as a preprocessing step for predictive models
- Different runs may produce different visualizations
- Distances between clusters in the final plot should not be interpreted literally

---

# Summary

| Algorithm | Learning Type | Primary Purpose |
|------------|---------------|-----------------|
| K-Means | Unsupervised | Partition data into K clusters |
| Hierarchical Clustering | Unsupervised | Build a hierarchy of clusters |
| DBSCAN | Unsupervised | Density-based clustering with outlier detection |
| Gaussian Mixture Models | Unsupervised | Probabilistic (soft) clustering |
| PCA | Unsupervised | Linear dimensionality reduction |
| t-SNE | Unsupervised | Nonlinear visualization of high-dimensional data |