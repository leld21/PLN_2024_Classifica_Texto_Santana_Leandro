from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import umap

# a)
data = pd.read_csv('Tweets_menor.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

# b)

print(f"Colunas do dataset: {data.columns}")
print(f"Classes: {data['sentiment'].unique()}")


# c)
print("5 primeiros registros")

print(train_data.head(5))

# d)
'''
class_distribution = data['sentiment'].value_counts()

class_distribution.plot(kind='bar')
plt.title('Distribuição das classes')
plt.xlabel('Classes')
plt.ylabel('Quantidade de registros')
plt.show()

'''
# e)

'''
data['text'] = data['text'].fillna('').astype(str)
data['text_length'] = data['text'].apply(len)

plt.figure(figsize=(10, 6))
data.boxplot(column='text_length', by='sentiment', grid=False, showfliers=False, widths=0.7)
plt.title('Tamanho dos Registros por Sentimento')
plt.suptitle('')  
plt.xlabel('Sentimento')
plt.ylabel('Número de Caracteres')
plt.show()

'''
# f) 
print("teste1")

tokenizer = AutoTokenizer.from_pretrained('MarieAngeA13/Sentiment-Analysis-BERT')

# Função de tokenização
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_texts = train_data['text'].fillna('').astype(str).tolist()
val_texts = val_data['text'].fillna('').astype(str).tolist()
test_texts = test_data['text'].fillna('').astype(str).tolist()

# Tokenizar os dados
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
val_encodings = tokenizer(val_texts, padding=True, truncation=True, return_tensors="pt")
test_encodings = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")

print("teste2")
# g)




