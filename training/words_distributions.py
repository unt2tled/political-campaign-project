import pandas as pd
import numpy as np

class WordsDistributionClass:
    def __init__(self,input_file_path,output_file_path):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
    
    def initialize_data(self):
        self.df = pd.read_csv(self.input_file_path)
        all_words = {}
        self.df['freq'] = self.df.apply(lambda x:
                WordsDistributionClass.get_words_freq_in_text(x['text'],all_words),axis=1)
        for word in all_words.keys():
            if all_words[word] >= 10:
                self.df['freq_'+word] = self.df.apply(lambda x:
                    0 if word not in x['freq'].keys() else x['freq'][word],axis=1)
        del all_words
        del self.df['freq']

    def get_words_freq_in_text(text,all_words):
        freq = {}
        text = text.lower()
        text = text.replace(";","")
        text = text.replace(",","")
        text = text.replace(".","")
        words = text.split(" ")
        for word in words:
            if word not in all_words:
                all_words[word] = 0
            if word not in freq.keys():
                freq[word] = 1
                all_words[word] += 1
            else:
                freq[word] += 1
        return freq

    def save_output(self):
        self.df.to_csv(self.output_file_path)

wdc = WordsDistributionClass('tagging.csv','words_distributions.csv')
wdc.initialize_data()
wdc.save_output()
