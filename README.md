# Instruction Tuning for Aspect-Based Sentiment Analysis
This repository contains an example of the code used for my MSc Data Science dissertation project.  

Given a piece of text data; for example, from a review of a restaurant:  
> "the pizza was great, but the service was slow"

_aspect-based sentiment analysis_ (ABSA) extracts the aspects (topics), such as "pizza" and "service" and the corresponding sentiments, "positive" or "negative".

Identifying relationships between sentiments and topics in text is a challenging task requiring a model to capture meaning and relationships across parts of a sentence. When the sentiment is not conveyed using an explicit adjective, e.g. "we were waiting for an hour for our food" (negative), or complex sentences describe multiple aspects and diverse sentiments, the task becomes especially challenging. 

My solution uses a pre-trained language model (Flan-T5) instruction-finetuned on the ABSA task using small labelled datasets, using Hugging Face tools in python.

## Findings and Limitations
Literature over the past 2-3 years shows transformer-based language models such as BERT, BART, or T5, provide the most effective starting point for ABSA, as they carry most of the information about the English language needed to infer aspects and sentiments. 

Instruction finetuning Flan-T5, which has been pre-trained for effective zero-shot instruction prompting on a wide variety of tasks (including simple sentiment analayis) achieved an effective ABSA model with very few training examples (n<10,000). A small number of errors occured when sentiment was ambiguous, or when the training data had been labelled inconsistently.  

The availability of labelled ABSA data severly limits the suitability of my final model. The training data available were limited to the domain of restaurant reviews. When tested on data from outside this domain, the model struggled to identify unfamiliar aspects. 

While I attempted to mitigate overfitting with cross-validation, it's possible my model is still slightly over-fit to the trianing domain. A more robust effective model could easily be trained given a relatively small but domain-diverse labelled training dataset.


## Files
- **data_preprocessing.py:**  tools for building instruction based prompts and label-based output sequences
- **utils.py:**  utils for HuggingFace-based experiments
- **flan-t5-ase.ipynb:**  Format data for ABSA, and perform task-specific training and inference on [Flan-T5](https://huggingface.co/google/flan-t5-base).  
- **Datasets:**  MAMS: Multi-Aspect Multi-Sentiment [\[papers with code\]](https://paperswithcode.com/dataset/mams) [\[data\]](https://github.com/siat-nlp/MAMS-for-ABSA)  
- **parse_mams.py:**  convert raw text and labels from xml format to csv

## Notes
Parts of this code were developed from [InstructABSA](https://github.com/kevinscaria/InstructABSA)
