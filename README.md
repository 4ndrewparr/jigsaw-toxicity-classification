# jigsaw-toxicity-classification [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/4ndyparr/jigsaw-toxicity-classification/master)

Training and inference code from the models used for the **Kaggle** competition [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). A NLP competition sponsored by the Conversation AI team (founded by Jigsaw and Google) with state-of-the-art winning solutions that apply transfer learning from the most powerful NLP models available at the time. These were: **BERT** (released by Google in October 2018) and **GPT2** (released by Open AI in February 2019).  
  
My final model consisted of a weighted ensemble of bidirectional LSTM models and BERT and GPT2 pretrained models finetuned with the competition data. It finished in the **top 2%** (56th/2,646) with a Private Leaderboard score of 0.94466 (PL score of the winning model was 0.94734).  

![Diagram](https://github.com/4ndyparr/jigsaw-toxicity-classification/blob/master/ensemble-jigsaw.png)  

In order to be able to train such heavy models like BERT and GPT2 with only the computational
resources from Kaggle Kernels (9-hour GPU with 16GB RAM), I had to apply techniques such as
**Gradient Accumulation** and **Mixed-Precision Training**. Other ML techniques applied in
the models were: **Multi-Task Learning** and **Checkpoint Ensemble**.
  
The fine-tuning notebooks for BERT and GPT2, the training code of the base LSTM model, and the inference notebook of the ensemble that was submitted as a final solution are included in this repository.  

  
Kaggle Profile: https://www.kaggle.com/andrewparr
