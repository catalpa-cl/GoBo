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

One of the main points of interest for the paper was finding the right balance between data requirements and performance gain. Accordingly, our second experiment focused on how the personalization technique deals with smaller datasets. For all 40 available writing styles we computed the reducting in CER for a total of 53 different dataset sizes (all multiples of 10 from 10 to 530 samples). As a result we were able to show that it is possible to obtain decent reductions in error rates even when using smaller subsets of the original datasets.

### Experiment 3: Personalization or just Task Adaptation

At this stage, we were concerned that the improvements we observed weren't due to an improved understanding of the target writing style but the result of the model
adapting to the properties of the new dataset (e.g. artefacts of the scanning process, the segmentation method or the pens used for writing). In order to show the latter wasn't the case, we retrained the model on the samples provided by a single writer and evaluated it on the remaining writing styles in our dataset. Assuming the improvements were caused by the model adapting to our dataset and not to the target writing style, the resulting model should have shown reduced error rates even for the remaining writers. As this wasn't the case we attributed the improvements to the model getting more familiar to the writing style.

### Experiment 4: Domain-Specific Results
