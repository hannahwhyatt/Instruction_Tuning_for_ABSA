from datasets import Dataset
from datasets.dataset_dict import DatasetDict


class promptBuilder:
    def __init__(self, task):
        self.task = task

        # Define task for aspect-sentiment (pair) extraction
        if self.task == "ase":
            self.bos = """Task: Extract the aspect terms and their corresponding sentiment classes (positive, neutral or negative) from the following input:
            """

        # Define task for aspect-opinion-sentiment (triplet) extraction
        elif self.task == "aose":
            self.bos == """Task: Extract the aspect terms, corresponding opinion terms, and corresponding sentiment classes (positive, neutral or negative) from the following input:
            """

        self.eos = "\noutput: "

    def generate_inputs(self, examples):
        """ Formate model input prompts: concatenate prompt bos and eos with raw text examples
        """
        inputs = [self.bos + example + self.eos for example in examples]
        return inputs
    
    def generate_target_outputs(self, **kwargs):
        """ Concatenate sentiment elements in to a list of labels for each instance.
        """
        target_outputs = []
        
        # for each instance
        for i in range(len(kwargs['aspects'])):
            label_list = ''
            
            # for each pair or triplet label in the instance
            for j in range(len(kwargs['aspects'][i])):
                a = kwargs['aspects'][i][j]
                s = kwargs['sentiments'][i][j]
                if self.task == "aste":
                    o = kwargs['opinions'][i][j]
                    label = a + ":" + o + ":" + s 
                elif self.task == "ase":
                    label =  a + ":" + s
                if len(label_list) == 0:
                    label_list = label_list + label
                else: 
                    label_list = label_list + ', ' + label
            target_outputs.append(label_list)
        
        return target_outputs

def create_hf_dataset(tr_input, tr_output, te_input, te_output, val_input, val_output):
    """ Create HuggingFace dataset with the train/test/val input and ouput data
    """
    train = Dataset.from_dict({'text': tr_input, 'labels': tr_output})
    test = Dataset.from_dict({'text': te_input, 'labels': te_output})
    val = Dataset.from_dict({'text': val_input, 'labels': val_output})
    dataset = DatasetDict({'train': train, 'test': test, 'validation': val})
    return dataset

