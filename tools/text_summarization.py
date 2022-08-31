"""
This module is for text summarization
"""
# ref: https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import pandas as pd
import numpy as np
import networkx as nx

class SummarizationClass:
    def read_text(text):
        text = text.replace("\"","")
        article = text.split(". ")
        sentences = []

        for sentence in article:
            #print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        #sentences.pop() 
        
        return sentences

    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []
    
        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]
    
        all_words = list(set(sent1 + sent2))
    
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
    
        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1
    
        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1
    
        return 1 - cosine_distance(vector1, vector2)
    
    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = SummarizationClass.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix


    def generate_summary(text, top_n=5):
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  SummarizationClass.read_text(text)
        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = SummarizationClass.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        #print(sentence_similarity_graph)
        try:
            scores = nx.pagerank(sentence_similarity_graph)

            # Step 4 - Sort the rank and pick top sentences
            ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)      

            for i in range(top_n):
                summarize_text.append(" ".join(ranked_sentence[i][1]))
        except nx.exception.PowerIterationFailedConvergence:
            print(f'text={text} was bad for nx')
            return ''
        # Step 5 - Offcourse, output the summarize texr
        return ". ".join(summarize_text)

class SummarizationClassRun:
    ''' class for running the summarization class algorithm with given parameters '''
    def __init__(self,input_file_path,text_column,output_file_path_keep_original_text_column):
        self.input_file_path = input_file_path
        self.text_column = text_column
        self.output_file_path_keep_original_text_column = output_file_path_keep_original_text_column
        self.output_file_path_override_text_column = output_file_path_override_text_column
    def run(self):
        # read input file as a dataframe
        df = pd.read_csv(self.input_file_path,encoding='cp1255')
        # add column with summarization of the text in the text column
        df['summarized_text'] = df[self.text_column].apply(lambda x: SummarizationClass.generate_summary(x, 1))
        # export output with the original text column to CSV file
        df.to_csv(self.output_file_path_keep_original_text_column,encoding='cp1255',index=False)
        # override original text column
        df[self.text_column] = df['summarized_text']
        del df['summarized_text']
        # export output with the overridden text column to CSV file
        df.to_csv(self.output_file_path_override_text_column,encoding='cp1255',index=False)
