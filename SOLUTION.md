# Hallucination Detection Solution

## 1. Reproducibility instructions

The solution is fully reproducible with the original repository structure.

Only the following files were modified:

- `aggregation.py`
- `probe.py`
- `splitting.py`

## Environment

The solution uses the provided language model:

```text
Qwen/Qwen2.5-0.5B
````

The model is loaded by the original `model.py`.
The final run was performed on NVIDIA Tesla T4 in Google Colab.

## Commands

To reproduce `results.json` and `predictions.csv`, run:

```bash
git clone https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION.git
cd SMILES-HALLUCINATION-DETECTION

pip install -r requirements.txt
python solution.py
```

## 2. Final solution

The final solution consists of three parts:

1. multi-layer hidden-state aggregation
2. compact geometrical and topological-inspired scalar features
3. ensemble probe based on PCA and logistic regression

## 2.1 Hidden-state aggregation

The baseline approach used only the last real token from the final transformer layer.
In the final solution, I use several later transformer layers:

```text
layers = 12, 16, 20, 24
```

For each selected layer, I focus on the tail of the sequence, because the input is `prompt + response`, and the final tokens are the most directly related to the generated answer.

For each selected layer I extract:

* the last real token representation;
* the mean representation over the last 64 real tokens;
* the standard deviation over the last 64 real tokens;
* the difference between the last token and the tail mean.

This gives the classifier information about both the final hidden state and the local behavior of the answer near the end of generation.

## 2.2 Geometrical features

In addition to raw hidden-state vectors, I add compact scalar features describing the geometry of the hidden-state trajectory.

For the tail-token cloud in each selected layer, I compute:

* mean distance of tokens to the centroid;
* standard deviation of distances to the centroid;
* maximum distance to the centroid;
* singular-value-based spectral features;
* effective rank;
* participation ratio.

These features describe whether the hidden states of the answer are concentrated, spread out, or dominated by a small number of directions.

I also compute trajectory features over consecutive token representations:

* total path length;
* mean step length;
* standard deviation of step length;
* maximum step length;
* average cosine similarity between consecutive steps;
* standard deviation of this cosine similarity.

The idea is that hallucinated and truthful answers may differ not only in the final hidden vector, but also in how the representation moves while the answer is generated.

## 2.3 Topological-inspired features

I also add a lightweight topological proxy based on minimum spanning tree statistics.

For a point cloud of tail-token hidden states, the 0-dimensional persistent homology death times correspond to the edge lengths of the minimum spanning tree. Instead of using an external persistent homology library, I compute MST edge statistics directly:

* mean MST edge length;
* standard deviation of MST edge lengths;
* maximum MST edge length;
* sum of MST edge lengths.

## 2.4 Probe classifier

The labelled dataset is small: 689 examples.
Because of this, a neural MLP probe can easily overfit high-dimensional hidden-state features.

The final probe uses a more stable classical pipeline:

```text
StandardScaler -> PCA -> LogisticRegression
```

Instead of using only one PCA dimension and one logistic regression model, I use an ensemble.

The PCA dimensions are:

```text
16, 32, 64, 96, 128
```

The logistic regression regularization strengths are:

```text
C = 0.03, 0.1, 0.3, 1.0
```

For each configuration, I train models with both:

```text
class_weight = None
class_weight = "balanced"
```

The final probability is the average predicted probability over all PCA + LogisticRegression models.

This ensemble makes the probe less sensitive to one specific PCA dimension or regularization strength.

## 2.5 Threshold selection

The probe outputs probabilities for the hallucinated class.

For validation folds, the decision threshold is selected to maximize validation accuracy.

For the final model used to produce `predictions.csv`, the threshold is chosen using the training positive-class prior. This is useful because the dataset is imbalanced:

```text
483 hallucinated / 206 truthful
```

So the positive-class rate is approximately:

```text
70.1%
```

The final `predictions.csv` has a similar class distribution:

```text
72 hallucinated predictions
28 truthful predictions
```

This prevents the final model from collapsing to an unrealistic class distribution.

## 2.6 Splitting strategy

The final evaluation uses stratified 5-fold splitting.
Each fold preserves the label distribution and has train, validation, and test parts.

This gives a more stable estimate than a single random split and helps detect overfitting.

## 3. Results

The final feature dimensionality is:

```text
14440
```

The results averaged over 5 folds are:

| Checkpoint                | Accuracy |     F1 |  AUROC |
| ------------------------- | -------: | -----: | -----: |
| Majority-class baseline   |   70.10% | 82.42% |    N/A |
| Probe on train split      |   79.70% | 86.92% | 91.71% |
| Probe on validation split |   75.96% | 84.41% | 74.92% |
| Probe on test split       |   75.04% | 83.94% | 75.59% |

The final probe improves the main accuracy metric over the majority-class baseline:

```text
75.04% - 70.10% = +4.94 percentage points
```

Per-fold test results:

| Fold | Test accuracy | Test F1 | Test AUROC |
| ---: | ------------: | ------: | ---------: |
|    1 |        76.09% |  85.46% |     80.81% |
|    2 |        75.36% |  84.82% |     76.41% |
|    3 |        75.36% |  84.55% |     76.49% |
|    4 |        73.91% |  81.44% |     72.69% |
|    5 |        74.45% |  83.41% |     71.52% |

Additional run information:

```text
Number of labelled samples: 689
Number of folds: 5
Feature extraction time: about 198 seconds
Final prediction distribution: 72% hallucinated, 28% truthful
```

## 4. Experiments and failed attempts

## 4.1 Last-token final-layer baseline

The initial solution used only the last real token from the final transformer layer.
This was simple and fast, but it did not use enough information from the hidden states. The resulting classifier was close to the majority-class baseline.

## 4.2 MLP probe

The original template used a small MLP trained with binary cross-entropy.
This approach can fit the training set, but the dataset is small and the feature space is high-dimensional. MLP was more prone to overfitting and was less stable on validation and test folds.

Because of this, I switched to a regularized PCA + LogisticRegression probe.

## 4.3 Global mean pooling over the full sequence

I also considered using mean pooling over the whole input sequence.
This was discarded because the sequence contains both the prompt and the response. Global mean pooling mixes the context and the generated answer too strongly.

The final solution focuses on the last 64 real tokens, which are more directly connected to the response.

## 4.4 Single PCA + LogisticRegression model

A single PCA + LogisticRegression model was more stable than the MLP, but the result depended on the chosen PCA dimensionality and regularization strength.

The final ensemble over several PCA dimensions, regularization strengths, and class-weighting modes gave better and more stable results.

## 4.5 Geometrical and topological-inspired features

I tried adding compact geometrical and topological-inspired features.
These features slightly improved the average test accuracy:

```text
Without geometrical/topological proxy: 74.45%
With geometrical/topological proxy:    75.04%
```

The improvement is small, but it is positive and comes with only a small increase in feature dimensionality:

```text
14359 -> 14440
```

Therefore, I kept these features in the final solution.

### 4.6 Factual bottleneck layers 12–15

I also tested an alternative aggregation inspired by the factual-bottleneck hypothesis.  
The idea was to use layers 12–15, with max-pooling on layers 12–13 and mean-pooling on layers 14–15. I also used compact geometric features such as vector norms, cosine jumps, residual gaps, and global semantic shift features.

This version produced a smaller feature vector:

```text
feature_dim = 3603
````

However, it did not improve the final metric:

```text
test accuracy = 73.59%
test F1       = 82.39%
test AUROC    = 75.14%
```

The model also showed stronger overfitting:

```text
train AUROC = 98.55%
test AUROC  = 75.14%
```

Therefore, I did not keep this version in the final solution.

### 4.7 EOS stripping

I also tested removing the last real token from the sequence before aggregation, assuming that it may correspond to an EOS or structural token.

This did not help in my setup. The result was:

```text
test accuracy = 73.30%
test F1       = 81.14%
test AUROC    = 75.21%
```

This was worse than the final version without EOS stripping:

```text
test accuracy = 75.04%
test F1       = 83.94%
test AUROC    = 75.59%
```

A likely reason is that my aggregation does not have exact access to the prompt/response boundary. Removing the last token may remove useful summary information from the full `prompt + response` representation. Because of this, the final solution keeps the last real token.
