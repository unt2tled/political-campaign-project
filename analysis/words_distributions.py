import pandas as pd
import numpy as np

class WordsDistributionClass:
    ''' This class is for creating a dataframe with the frequencies
        of the words in the text column of the input file, in addition
        to the file's original columns. '''
    def __init__(self,input_file_path,output_file_path,text_column='text'):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.text_column = text_column
    def initialize_data(self):
        # read dataframe from the input CSV file path
        self.df = pd.read_csv(self.input_file_path,encoding='cp1255')
        # add frequencies of the words in the text column as columns
        # for the dataframe which was previously read
        # Impl. Note: all_words is a dictionary for the words' frequencies
        #             to be used during the calculation. It's a local variable.
        #             for word in all_words.keys():
        #             all_words[word] == # videos which contain word
        #                                   as part of the text in in the text column
        all_words = {}
        self.df['freq'] = self.df.apply(lambda x:
                WordsDistributionClass.get_words_freq_in_text(x[self.text_column],all_words),axis=1)
        for word in all_words.keys():
            if all_words[word] >= 10:
                self.df['freq_'+word] = self.df.apply(lambda x:
                    0 if word not in x['freq'].keys() else x['freq'][word],axis=1)
        del all_words
        del self.df['freq']

    def get_words_freq_in_text(text,all_words):
        # static public function
        freq = {}
        # our calcuation is not sensitive to CAPS-LOCK characters
        text = text.lower()
        # our calcuation is not sensitive to the characters: ";",",","."
        # NOTE: we are sensitive to other characters, including question marks
        # and '"', "'" etc.
        text = text.replace(";","")
        text = text.replace(",","")
        text = text.replace(".","")
        words = text.split(" ")
        # algorithm for assigning words distribution
        # for given all_words dictionary
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
        #export dataframe to output CSV file path
        self.df.to_csv(self.output_file_path,index=False)
