# jigsaw-toxicity-classification [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/4ndyparr/jigsaw-toxicity-classification/master)

Training and inference code from the models used for the **Kaggle** competition [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). A NLP competition sponsored by the Conversation AI team (founded by Jigsaw and Google) with state-of-the-art winning solutions that apply transfer learning from the most powerful NLP models available at the time. These were: **BERT** (released by Google in October 2018) and **GPT2** (released by Open AI in February 2019).  
  
My final model consisted of a weighted ensemble of bidirectional LSTM models and BERT and GPT2 pretrained models finetuned with the competition data. It finished in the **top 2%** (56th/2,646) with a Private Leaderboard score of 0.94466 (PL score of the winning model was 0.94734).  

![Diagram](https://github.com/4ndyparr/jigsaw-toxicity-classification/blob/master/ensemble-jigsaw.png)  

In order to be able to train such heavy models like BERT and GPT2 with only the computational
resources from Kaggle Kernels (9-hour GPU with 16GB RAM), I had to apply techniques such as
**Gradient Accumulation** and **Mixed-Precision Training**. Other ML techniques applied in
the models were: **Multi-Task Learning** and **Checkpoint Ensemble**.
  
The fine-tuning notebooks for BERT and GPT2, the training code of the base LSTM model, and the inference notebook of the ensemble that was submitted as a final solution are included in this repository.  

![Diagram](https://github.com/4ndyparr/jigsaw-toxicity-classification/blob/master/LSTM1-jigsaw.png) 

Kaggle Profile: https://www.kaggle.com/andrewparr

## Keys of the Competition

### Evaluation Metric

The focus of this competition was on minimizing the bias that toxicity models (with ***toxicity*** defined as: *anything rude, disrespectful or otherwise likely to make someone leave a discussion*) usually account for. These models tend to learn incorrectly to associate the presence of frequently attacked identities like *gay* or *black* to toxicity, even in non-toxic contexts.

Consequently, a customized metric was defined to rank the models of this competition. Basically, to the overall AUC they added three bias-focused AUCs, as explained [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation), to compute the score. It can be written like this:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?score%20=%20w_0%20AUC_{overall}%20+%20\sum_{i=1}^{3}%20w_i%20AUC_{bias_{i}}">
</p>  
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?w_0,w_1,w_2,w_3}%20=%20{0.25}">
</p>  



### The Loss Function

Being the official metric too computationally expensive to use directly as loss function, designing an efficient loss that still focuses on minimizing identity bias was critical in this competition.

#### Weighted BCE
The most common approach (and the one I took) was to use some kind of *weighted Binary Cross Entropy*, with somehow larger weights on samples that mention the identities. After some hyperparameter tuning, I ended up using the following formula:

```python
sample_weights = np.ones(len(train_df), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
```

#### Multi-Task Learning
Training the network with auxiliary targets (from additional info found in the train samples) also proved to be very effective. The network learns to predict these other targets (I used up to 5 **auxiliary targets**: *'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'*) and how the presence of these can translate into the main target *'toxicity'*, thus helping it make more accurate predictions.
```python
Y_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
```

#### Confidence Score
One of my original contributions in the competition was to further modify the weights to reflect the 'confidence' in the labeling. 
The samples had been manually labeled and one of the features in the train dataset was *'toxicity_annotator_count'*, specifying the number of times that a sample had been labeled (the final labels were just an average of these individual annotations).
After observing that my models' biggest mistakes (*error = prediction - target*) were happening on mislabelled samples with the minimum annotator count I decided to modify the weights accordingly.
This gave all my models a nice boost, since they were now paying less attention to noisy, misleading samples.
Since this tweak wasn’t publicly shared during the competition, it gave an edge to my models. When the competition ended, and a lot of solutions were published, I found out that only one other participant had come up with something similar.
```python
train_df['ta_count'] = train_df['toxicity_annotator_count'] / train_df['toxicity_annotator_count'].mean()
sample_weights *= train_df['ta_count_coef']
```

### Ensemble

With external sources allowed in the competition, it became soon obvious that the winning solutions were going to be ensembles including fine-tunnings of state-of-the-art NLP models such as BERT or GPT2 (***Transfer Learning***). LSTM architectures were outperformed by these models but they were ensembling very well with them.

Thus, the task at hand was to create a variety of models from these architectures and pushing each one as far as possible to make the final ensemble stronger.

### Submission Constraints

The submission kernel had a limited runtime (120'), a limited RAM (13GB) and a maximum of 20GB of database storage (each trained model used for inference was uploaded as a database). Balancing these constraints while at the same time considering how much stronger each particular model is expected to make the ensemble was a complex task. 

My submission ensemble included:
- 5 BERT models
-	3 GPT2 models
-	6 LSTM1 models
-	5 LSTM2 models





  

