from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from deep_translator import GoogleTranslator

"""
# Welcome to Product Categorization Demo!
Hi Cornershop!
Upload your data in csv format to predict category and sub-category.
"""


with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))

    import numpy as np
import csv
import re
import torch



def remove_bracket2(sample_str):
    clean4 = re.compile('\(.*?\)')
    cleantext = re.sub(clean4, ' ', sample_str)
    return cleantext.strip()

def remove_comma(sample_str):
    clean4 = re.compile(',')
    cleantext = re.sub(clean4, '', sample_str)
    return cleantext.strip()

def remove_period(sample_str):
    clean5 = re.compile('\\.')
    cleantext = re.sub(clean5, '', sample_str)
    return cleantext.strip()

def special_characters(sample_str):
    alphanumeric = ""
    for character in sample_str:
        if character.isspace():
            alphanumeric += character
        elif character.isalnum():
            alphanumeric += character
    return alphanumeric.strip()


# st.title('Product Categorization')
# st.write('Hello Cornershop!')


# data = data.replace(np.nan,'',regex=True) 
# un_lbl = data['Sub-Category-en'].str.lower().unique()

# np.save('sub_cat_list.npy',un_lbl,allow_pickle=True)
un_lbl  = np.load('/home/kpraja4/kpraja4_nfs/Product_Categorization/Sttreamlit/sub_cat_list.npy',allow_pickle=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.texts = [tokenizer(str(text), 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df]
        
    
    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    
    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        
        return batch_texts

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 136)
        self.sigm = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigm(linear_output)

        return final_layer

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')    


model = BertClassifier()
model.load_state_dict(torch.load('/home/kpraja4/kpraja4_nfs/Product_Categorization/Sttreamlit/4_model.pth',map_location=torch.device('cpu')))

in_datat = st.file_uploader("Choose a file")
if in_datat is not None:
    data = pd.read_csv(in_datat)
    data['item_name'] = data['item_name'].str.replace('[^\w\s]','')
    data = data.replace(np.nan,'',regex=True) 
    ind = [i for i in range(len(data)) if len(data['item_name'][i])>0]
    clean_data = [data['item_name'][i] for i in ind]
    st.write('Step-1/3: Translating data into English...')
    eng_data1 = GoogleTranslator(source='auto', target='en').translate_batch(clean_data)
    st.write('Step-2/3: Predictting category and sub-category...')
    test = Dataset(eng_data1)
    test_dataloader = torch.utils.data.DataLoader(test)
    t_pred=[]
    with torch.no_grad():
        for test_input in test_dataloader:
            mask = test_input['attention_mask']
            input_id = test_input['input_ids'].squeeze(1)
            output = model(input_id, mask)
            t_pred.append(output.argsort(dim=1,descending=True)[:,:2].numpy())
    pred = [un_lbl[ind] for ind in t_pred]
    st.write('Step-3/3: Preparing prediction result to download...')
    endata = []
    pre_d1 = []
    pre_d2 = []
    k=0
    for i in range(len(data)):
        if i in ind:
            endata.append(eng_data1[k])
            pre_d1.append(pred[k][0][0])
            pre_d2.append(pred[k][0][1])
            k+=1
        else:
            endata.append('')
            pre_d1.append('')
            pre_d2.append('')
    data['English_Translated_Item'] = endata
    data['Prediction-1'] = pre_d1
    data['Prediction-2'] = pre_d2
    csv = convert_df(data)

#     st.download_button("Press to Download",csv,key='download-csv')
    st.dataframe(data=data)
else:
    st.write('Upload your data to predict')
    st.write('It must be in the csv file and data column required to named as \'item_name\'')
