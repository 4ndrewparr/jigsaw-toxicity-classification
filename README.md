# jigsaw-toxicity-classification [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/4ndyparr/jigsaw-toxicity-classification/master)



Training and inference code from the models used for the **Kaggle** competition [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). A NLP competition sponsored by the Conversation AI team (founded by Jigsaw and Google) with state-of-the-art winning solutions that apply transfer learning from the most powerful NLP models available at the time. These were: **BERT** (released by Google in October 2018) and **GPT2** (released by Open AI in February 2019).  
  
My final model consisted of a weighted ensemble of bidirectional LSTM models and BERT and GPT2 pretrained models finetuned with the competition data. It finished in the **top 2%** (56th/2,646) with a Private Leaderboard score of 0.94466 (PL score of the winning model was 0.94734).  

![Diagram](https://github.com/4ndyparr/jigsaw-toxicity-classification/blob/master/ensemble-jigsaw.png)  

In order to be able to train such heavy models like BERT and GPT2 with only the computational
resources from Kaggle Kernels (9-hour GPU with 16GB RAM), I had to apply techniques such as
**Gradient Accumulation** and **Mixed-Precision Training**. Other ML techniques applied in
the models were: **Multi-Task Learning** and **Checkpoint Ensemble**.
  
The fine-tuning notebooks for BERT and GPT2, the training code of the base LSTM model, and the inference notebook of the ensemble that was submitted as a final solution are included in this repository.  

![Diagram](https://github.com/4ndyparr/jigsaw-toxicity-classification/blob/master/LSTM-jigsaw.png) 


## Keys of the Competition

### Evaluation Metric

The focus of this competition was on minimizing the bias that toxicity models (with ***toxicity*** defined as: *anything rude, disrespectful or otherwise likely to make someone leave a discussion*) usually account for. These models tend to learn incorrectly to associate the presence of frequently attacked identities like *gay* or *black* to toxicity, even in non-toxic contexts.

Consequently, a customized metric was defined to rank the models of this competition. Basically, the score is the average of the overall AUC and three bias-focused AUCs, as explained in detail [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation).  

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?score%20=%20w_0%20AUC_{overall}%20+%20\sum_{i=1}^{3}%20w_i%20AUC_{bias_{i}}">
</p>  
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?w_0,w_1,w_2,w_3}%20=%20{0.25}">
</p>  



### The Loss Function

Being the official metric too computationally expensive to use directly as loss function, designing an efficient loss that still focuses on minimizing identity bias was critical in this competition.

#### Weighted BCE
The most common approach (and the one I took) was to use some kind of *weighted Binary Cross Entropy*, with somehow larger weights on samples that mention the identities. The algorithm to determine these weights could be tuned as another hyperparameter. This is one of the formulas that I ended up using:

```python
sample_weights = np.ones(len(train_df), dtype=np.float32)
sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
```

#### Multi-Task Learning
Training the network with auxiliary targets (from additional info found in the train samples) also proved to be very effective. The network learns to predict these other targets (I used up to 5 **Auxiliary Targets**: *'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat'*) and how the presence of these can translate into the main target *'toxicity'*, thus helping it make more accurate predictions.

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
sample_weights *= train_df['ta_count']
```

### Ensembling

With external sources allowed in the competition, it became soon obvious that the winning solutions were going to be ensembles including fine-tunings of state-of-the-art NLP models such as BERT or GPT2 (**Transfer Learning**). LSTM architectures were outperformed by these models but they were ensembling very well with them.

Thus, the task at hand was to create a variety of models from these architectures and pushing each one as far as possible to make the final ensemble stronger.

Seeking model diversity, I worked on two variants of the general LSTM architecture shown in the diagram, LSTM1 and LSTM2. The main differences between these two, besides one being written in Keras and the other in PyTorch (and using some Fast AI libraries), are:
- A different *Text Preprocessing*  

- *Spatial Dropout* of 0.2 for LSTM1, 0.3 for LSTM2
- *Toxicity Target* kept as a *continuous* variable in LSTM1, while converted to *binary* (0,1 values) in LSTM2
- 5 *Auxiliary Targets* in LSTM1, but 6 in LSTM2 (toxicity target is also included as aux. target)
- An slightly different *Loss Function* (different weights algorithm)
- *Sequence Bucketing* applied in LSTM2

### Rebuilding the Embedding Matrix

At inference time, LSTM loaded models (which include the embedding matrices) will be tested with a new dataset. What do we do with the embedding matrix? We have different options:

#### The Good Solution

We can keep the old embedding matrix if we also use the old tokenizer. The tokenizer will be just not giving a token to new words that may appear in the test set, so the model will work, but the information from these new words will be lost. Since the train set is considerably larger, there should not be too many new words anyways.

#### The Better Solution

The more words with embeddings, the more info we are feeding to the model, this is why rebuilding the embedding matrix with the words from the test set is the optimal solution.

To do this we need to first build an oversized new embedding matrix filled with zeros when you run out of words (the test set will contain less words that the train set and to be able to swap the old matrix for the new one, they need to have the same size ```(N_EMBS_LSTM, 600)```).
```python
def build_matrix_new(word_index, path):
    embedding_index = load_embeddings_new(path)
    embedding_matrix = np.zeros((N_EMBS_LSTM, 300))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            pass
    return embedding_matrix
    
emb_matrix_new = np.concatenate(
    [build_matrix_new(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)    
```

And then we will able to swap this matrix for the old one by selecting the embeddings layer and setting the weights (the embeddings):
```python
model = load_model(os.path.join(model_path, checkpoint))
model.layers[1].set_weights([embedding_matrix_new])
```

This embedding matrix will have embeddings for every word in the test set, and the model's accuracy should benefit from this!



### Submission Constraints

The submission kernel had a limited runtime (120'), a limited RAM (13GB) and a maximum of 20GB of database storage (each trained model used for inference was uploaded as a database). Balancing these constraints while at the same time considering how much stronger each particular model is expected to make the ensemble was a complex task. 

Subensembles of BERTs, my strongest models, were reaching an elbow at around n=4/n=5, to the point that when n=10, they were giving a similar performance.

<p align="center">
  <img src="https://github.com/4ndyparr/jigsaw-toxicity-classification/blob/master/BERT-subensemble.png" height="400">
</p>  
<p align="center">
 

Consequently I went for 5 BERT models. This left time to fit at most 4 GPT2 models (the second strongest model), but to avoid getting into runtime trouble I settled for 3 GPT2 models. I filled the rest of the size and time space with subensembles of both LSTM variants (rebuilding the embedding matrix adds ~ 30" to the preprocessing LSTMs preprocessing times shown).

architecture|n models|prepr. time|infer. time|total time|model size|total size
:---:|:---:|---:|---:|---:|---:|---:
BERT|5|120"|725"|3,745"|1.22GB|6.10GB
GPT2|3|45"|800"|2,445"|1.44GB|4.32GB
LSTM1|6|10"|25"|160"|0.77GB|4.62GB
LSTM2|5|35"|20"|135"|0.97GB|4.85GB
TOTAL|19|||6,485"||19.89GB
**MAX**||||**7,200"**||**20.00GB**




Kaggle Profile: https://www.kaggle.com/andrewparr



  

