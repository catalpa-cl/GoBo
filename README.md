# GoBo a Handwriting Recognition dataset for Personalization


This repository comprises the collected dataset of handwritten transcriptions of the ASAP SAS.  

## Terms of Use & Citation
This database may be used for non-commercial research purpose only. 
If you publish material based on this database - please refer to the following paper:

> Christian Gold, Dario van den Boom, Torsten Zesch. 2021. Personalizing Handwriting Recognition Systems with Limited User-Specific Samples. 16th International Conference on Document Analysis and Recognition (ICDAR).


## Dataset Statistics (v.1.0)
The latest version of the dataset can be downloaded [here](link missing).

40 Participants
5 sets from different sources for personalization 
2 sets from 2 domains for testing
926 words/writer, 37k words in total

Gender & Age
| female | male | 20-29 | 30-39 | 40-49 | 50-59 | 60-99 | avg |
| ------ | ---- | ----- | ----- | ----- | ----- | ----- | --- |
|     17 |   23 |    22 |    10 |     1 |     5 |     2 |  34 |


## Experiments

### Experiment 1: Impact of Personalization

In this experiment we evaluated the effectiveness of the chosen personalization technique by comparing the baseline model's performance before and after being retrained on all training samples provided by a writer. We then averaged the results across all participants and found the technique to reduce the model's CER on all test sets from 14.1% to 8% on average. Encouraged by the results we then chose to proceed with improving the data requirements in the following experiments.

### Experiment 2: Required Amount of Personalized Handwriting Samples
