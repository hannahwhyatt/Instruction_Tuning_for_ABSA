# instruction_tuning
Tools for instruction fine-tuning a language model via HuggingFace.  
Available:
- ABSA: (aspect-sentiment (pair) extraction)  


## General
- **data_preprocessing.py:**  tools for building instruction based prompts and label-based output sequences
- **utils.py:**  utils for HuggingFace-based experiments

## Aspect-Based Sentiment Analysis (ABSA)
- **flan-t5-ase.ipynb:**  Format data for ABSA, and perform task-specific training and inference on [Flan-T5](https://huggingface.co/google/flan-t5-base).  
- **Datasets:**  MAMS: Multi-Aspect Multi-Sentiment [\[papers with code\]](https://paperswithcode.com/dataset/mams) [\[data\]](https://github.com/siat-nlp/MAMS-for-ABSA)  
- **parse_mams.py:**  convert raw text and labels from xml format to csv

## Notes
Parts of this code were developed from [InstructABSA](https://github.com/kevinscaria/InstructABSA)
