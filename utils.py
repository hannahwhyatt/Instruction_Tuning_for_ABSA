import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)


class instructionModelHandler:
    def __init__(self, base_model, task=None):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')
        self.task = task

    def get_sentiment_analysis_metrics(self, y_true, y_pred, task):
        """ Parse labels from output strings and calculate precision, recall, and macro f1.
        """
        total_pred = 0
        total_gt = 0
        tp = 0

        if task == "ase":
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp+=1
                            break
        # WIP
        elif task == "aose":
            pass

        p = tp/total_pred
        r = tp/total_gt

        return p, r, 2*p*r/(p+r)
         
    def compute_metrics_fn(self, eval_pred):
        """ Compute metrics function to pass to HF Seq2SeqTrainer.
        Parse output strings and generate precision, recall and f1 score.
        """
        eval_pred_seq_tkns, eval_true_seq_tkns = eval_pred
        eval_pred_seq_tkns = [[p for p in l if p != 0] for l in eval_pred_seq_tkns]
        eval_true_seq_tkns = [[p for p in l if p != -100] for l in eval_true_seq_tkns]        
        eval_pred_seq = [self.tokenizer.decode(instance, skip_special_tokens=True) for instance in eval_pred_seq_tkns]
        eval_true_seq = [self.tokenizer.decode(instance, skip_special_tokens=True) for instance in eval_true_seq_tkns]

        if self.task in ['ase', 'aose']:
            metrics_function = self.get_sentiment_analysis_metrics
        else:
            return
        
        eval_p, eval_r, eval_f1 = metrics_function(eval_true_seq, eval_pred_seq, task='ase')
        metrics = {'eval precision':eval_p, 'eval recall':eval_r, "eval f1":eval_f1}
        return metrics
    
    def train(self, tokenized_datasets, **kwargs):
        """ HF Seq2SeqTrainer object to train the model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        self.model.train()

        # Define trainer object
        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics_fn
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        trainer.train()
        self.model.save()
        return trainer
    
    def generate_output(self, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'train'):
        """ Inference on a subset of the data
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)
        self.model.eval()

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
                
        return predicted_output
