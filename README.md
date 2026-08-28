![Image alt text](/assets/biomQuants_Logo_v1.png)
# biomQuants

![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![Repo](https://img.shields.io/badge/github-repo-yellow?logo=github)
![Project repo](https://img.shields.io/badge/Project-repo-green)

An open source package for biometric quantification: quantifiers and evaluation measures.

The package consists of several quantifiers:

1. DGBQA
2. Delta Distance
3. MasterFace
4. Generative Capacity
5. Swipe Quality

We provide several measures:

1. Rank deviation ($\hat{r}$)
2. Relevance ($\mathcal{R}$)
3. Trend match distance ($\Psi$)
4. ICGD Score ($C_d$)
5. Advanced acceptance score (${A_r}^*$)
6. Standard evaluation measures:    

## Requirements

1. numpy
2. sckit-learn
3. scipy
4. tensorflow $\geq$ 2.8.0

## How to use

### Quantifiers
```python
from biomQuants.qauntifers import getScores
scores = getScores(embPath='Path to embeddings',
                    quantifier=quantifier,
                    y_cat=labels,
                    y_id=idLabels,
                    G_total=numCategories,
                    I_total=numIdentities)                    
```
Choice of quantifiers: ['dgbqa','deltaDistance','masterFace','genCapacity','swipeQuality'] 

y_cat: Category labels

y_id: Identity labels

### Evaluation measures
1. Advanced Acceptance Score

```python
from biomQuants.advancedAcceptance import comp_advancedAcceptance
nAr_star = comp_advancedAcceptance(scores,
                                   groundTruth,
                                   embeddings,
                                   labels,
                                   G=numCategories)
```

2. Rank deviation

```python
from biomQuants.rankDev import rankDev
r_prime = rankDev(1-groundTruth,
                  scores,
                  G=numCategories)
```

3. Relevance

```python
import numpy as np
from biomQuants.acceptanceScore import compAr

def preProcess(inputVec):
    inputVec = (inputVec - np.mean(inputVec))/np.std(inputVec)
    return inputVec/np.linalg.norm(inputVec)

relevance = compAr(preProcess(scores),
                   preProcess(groundTruth),
                   normalizer=False,
                   relevance=True)
```

4. ICGD score

```python
import tensorflow as tf
from biomQuants.icgd import compICGD

def normalisation_layer(x):   
    return(tf.math.l2_normalize(x, axis=1, epsilon=1e-12))

embeddings = tf.keras.layers.Lambda(normalisation_layer)(embeddings)

icgdScore = icgdScore(embeddings.numpy(),
                     labels)
```

5. Trend match distance

```python
from biomQuants.trendMatch import compTrendMatchDist
psi = rankDev(scores,
              groundTruth,
              G=numCategories)
```
