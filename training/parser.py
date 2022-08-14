import pandas as pd
import numpy as np

class TagFilesGenerator:
    def __init__(self, path_name, filter_definite, split_num, shorten_center_to_base_size, prioritize_americans,
                 filter_func):
        self.path_name = path_name
        self.filter_definite = filter_definite
        self.split_num = split_num
        self.shorten_center_to_base_size = shorten_center_to_base_size
        self.prioritize_americans = prioritize_americans
        self.filter_func = filter_func
        
    def split_texts(self):
        self.df['text'] = self.df['text'].str.replace('"','')
        self.df['sentences'] = self.df['text'].str.split('.')
        self.df['sentences'] = self.df['sentences'].apply(lambda x: [e.strip() for e in x])
        self.df['text1'] = self.df['sentences'].apply(lambda x:". ".join(x[:int(len(x)/self.split_num)])+".")
        self.df['copy']=0
        tmp_df = self.df.copy()
        tmp_i_df = None
        for i in range(1,self.split_num):
            tmp_i_df = tmp_df.copy()
            tmp_i_df['copy']=i
            tmp_i_df['text1'] = tmp_i_df['sentences'].apply(lambda x:".".join(x[int(i*len(x)/self.split_num):int((i+1)*len(x)/self.split_num)]))
            self.df = pd.concat([self.df,tmp_i_df],axis=0)
        self.df.reset_index(inplace=True)
        self.df['text'] = self.df['text1']
        self.df.sort_values(['name'],inplace=True)
        del tmp_i_df
        del self.df['text1']
        del self.df['sentences']
        
    def write_tag_files(self,split_texts):
        ''' requirements: run add_code_label_column_in_sign_version() '''
        #base model
        self.df['label'] = (self.df['code_label']>=0)+0
        self.df[['name','text','label']].to_csv('tags_base.csv' if not split_texts else 'tags_base_double.csv',index=False)
        #center model
        self.df['label'] = (self.df['code_label']<=0)+0
        self.df[['name','text','label']].to_csv('tags_center.csv' if not split_texts else 'tags_center_double.csv',index=False)
        #3Label mode
        self.df['label'] = self.df['code_label']+1
        self.df[['name','text','label']].to_csv('tags.csv' if not split_texts else 'tags_double.csv',index=False)
        
    def exec_shorten_center_to_base_size(self):
        base_size = len(self.df[self.df['code_label']==1])
        df_center = self.df.copy()
        self.df = self.df[self.df['code_label']!=-1]
        df_center = df_center[df_center['code_label']==-1]
        df_center = df_center.head(base_size) # could be altered to random choice of size base_size from df_center
        self.df = pd.concat((self.df,df_center)).reset_index()
        del self.df['index']
        
    def add_code_label_column_in_sign_version(self):
        ''' requirements: run map_taggings_to_signs() '''
        self.taggers = [tagger+"_int" for tagger in self.taggers]
        self.df['code_label'] = np.sign(self.df[self.taggers].sum(axis=1))
        if (self.prioritize_americans):
            self.df['code_label'] = np.where((self.df['Ben']!=2)*(self.df['Ben']==self.df['Skyler']), self.df['Ben'], self.df['code_label'])
            
    def exec_filter_definite(self):
        self.df = self.df[self.df.apply(lambda x: x['definite']!=3,axis=1)]
        
    def map_taggings_to_signs(self):
        ''' NOTE: if self.filter_definite then add definite column as well '''
        self.taggers = [c for c in self.df.columns.values if c!='name' and c!='text' and c!='text_ocr']
        if (self.filter_definite):
            self.df['definite'] = 0
        for tagger in self.taggers:
            self.df[tagger] = self.df[tagger].map({'not center':0,'base':1, 'Base':1,'both':0, 'base/both':0, 'center':-1,'Already Tagged':2,'0':2}).astype('int')
            self.df[tagger+'_int'] = self.df[tagger].apply(lambda x: 0 if x==2 else x)
            if (self.filter_definite):
                self.df['definite'] = self.df['definite'] | np.where(self.df[tagger+'_int'].isin([0,2]),0,2**((self.df[tagger+'_int']+1)/2)).astype('int')
                
    def run(self):
        self.df = pd.read_csv(self.path_name)
        if (self.filter_func != None):
            self.df = self.df[self.df.apply(self.filter_func)]
        # Keep videos with text alone
        self.df.dropna(subset=['text'], inplace=True)
        self.df = self.df[self.df.apply(lambda x:len(x['text'])>5,axis=1)]
        # Keep unique videos
        self.df.drop_duplicates('name',inplace=True)
        self.map_taggings_to_signs()
        if (self.filter_definite):
            self.exec_filter_definite()
        self.add_code_label_column_in_sign_version()
        if (self.shorten_center_to_base_size):
            self.exec_shorten_center_to_base_size()
        # Write tag files with no split
        self.write_tag_files(False)
        if (self.split_num>=2):
            self.split_texts()
            # Write tag files with split of size self.split_num
            self.write_tag_files(True)

if __name__ == "__main__":
    def name_filter_by_head(name: str,head_size: int,head_content: str) -> bool:
        return head_size<=0 or name[:head_size] == head_content
    
    tfg = TagFilesGenerator("data/tagging_new.csv", True, 1, True, False, None)
    tfg.run()
