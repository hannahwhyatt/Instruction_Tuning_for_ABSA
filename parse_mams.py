from xml.etree.ElementTree import parse
import pandas as pd

def parse_aspect_terms(path, lowercase=False, grouped=True):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    for sentence in sentences:
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        if grouped:
            terms = []
            for aspectTerm in aspectTerms:
                if lowercase:
                    terms.append(aspectTerm.get('term').lower())
                else:   
                    terms.append(aspectTerm.get('term'))
            data.append(terms)
        else:
            for aspectTerm in aspectTerms:
                term = aspectTerm.get('term')
                if lowercase:
                    term = term.lower()
                data.append(term)
    return data

def parse_polarities(path, grouped=True):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    for sentence in sentences:
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        if grouped:
            polarities = []
            for aspectTerm in aspectTerms:
                polarities.append(aspectTerm.get('polarity'))
            data.append(polarities)
        else:
            for aspectTerm in aspectTerms:
                polarity = aspectTerm.get('polarity')
                data.append(polarity)
    return data

def parse_sentences(path):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    for sentence in sentences:
        text = sentence.find('text').text
        if text is None:
            continue
        else:
            data.append(text)
    return data

root_path = './Datasets/MAMS/'

train_text = parse_sentences(root_path + 'train.xml')
test_text = parse_sentences(root_path + 'test.xml')
val_text = parse_sentences(root_path + 'val.xml')

train_aspects = parse_aspect_terms(root_path + 'train.xml')
test_aspects = parse_aspect_terms(root_path + 'test.xml')
val_aspects = parse_aspect_terms(root_path + 'val.xml')

train_sentiments = parse_polarities(root_path + 'train.xml')
test_sentiments = parse_polarities(root_path + 'test.xml')
val_sentiments = parse_polarities(root_path + 'val.xml')

train = pd.DataFrame({'raw_text': train_text, 'aspects': train_aspects, 'sentiments': train_sentiments})
test = pd.DataFrame({'raw_text': test_text, 'aspects': test_aspects, 'sentiments': test_sentiments})
val = pd.DataFrame({'raw_text': val_text, 'aspects': val_aspects, 'sentiments': val_sentiments})

train.to_csv(root_path + 'train.csv')
test.to_csv(root_path + 'test.csv')
val.to_csv(root_path + 'val.csv')