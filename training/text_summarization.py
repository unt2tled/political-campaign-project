# ref: https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70



from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import pandas as pd
import numpy as np
import networkx as nx

class SummarizationClass:
    def read_text(text):
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
        #print("before")
        #print(text)
        sentences =  SummarizationClass.read_text(text)
        #print("after")
        #print(sentences)
        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = SummarizationClass.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        print(sentence_similarity_graph)
        try:
            scores = nx.pagerank(sentence_similarity_graph)

            # Step 4 - Sort the rank and pick top sentences
            ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
            #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

            for i in range(top_n):
                #print(ranked_sentence)
                summarize_text.append(" ".join(ranked_sentence[i][1]))
        except nx.exception.PowerIterationFailedConvergence:
            print(f'text={text} was bad for nx')
            return ''
        # Step 5 - Offcourse, output the summarize texr
        #print("Summarize Text: \n", ". ".join(summarize_text))
        return ". ".join(summarize_text)


input_file_path = 'tagging.csv'
output_file_path = 'summarized_tagging.csv'

df = pd.read_csv(input_file_path,encoding='utf8')
df['summarized_text'] = df['text'].apply(lambda x: SummarizationClass.generate_summary(x, 1))
df.to_csv(output_file_path,encoding='utf8',index=False)