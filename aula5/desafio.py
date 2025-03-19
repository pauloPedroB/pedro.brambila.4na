import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer




# Baixar recursos necessários do NLTK
nltk.download('stopwords', download_dir='/home/codespace/nltk_data')
nltk.download('wordnet', download_dir='/home/codespace/nltk_data')
nltk.download('omw-1.4', download_dir='/home/codespace/nltk_data')  # Open Multilingual WordNet
nltk.data.path.append('/home/codespace/nltk_data')
nltk.download('punkt', quiet=False)
nltk.data.path.append('/home/codespace/nltk_data')


def basic_cleaning(text):
    # Converter para minúsculas
    text = text.lower()
    
    # Remover pontuações
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remover números
    text = re.sub(r'\d+', '', text)
    
    # Remover espaços extras
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokanize_text(text):
    """
    Tokeniza o texto em palavras
    """

    text = basic_cleaning(text)
  
    tokenizer = TreebankWordTokenizer()

    word_tokens = tokenizer.tokenize(text)
    
    
    return word_tokens



def remove_stopwords(tokens):
    """
    Remove stopwords da lista de tokens
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(frases):
    tokens_no_stopwords_total = []
    for text in frases:
        text_cleaned = basic_cleaning(text)
        tokens = tokanize_text(text_cleaned)
        tokens_no_stopwords = remove_stopwords(tokens)
        tokens_no_stopwords_total.extend(tokens_no_stopwords)
        
     
        lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens_no_stopwords_total]


frase1= "The children were playing in the leaves yesterday."
frase2= "She studies computer science and is taking three courses."
frase3= "The wolves howled at the moon while mice scurried in the grass."
frase4= "He was driving faster than the cars around him."
frase5= "The chefs used sharp knives to prepare the tastiest dishes."
frase6="running better studies wolves mice children was ate swimming parties leaves knives happier studying played goes driving talked"
frases = [frase1, frase2, frase3, frase4, frase5,frase6]

frase_lematizada = lemmatize_tokens(frases)
for token in frase_lematizada:
    print(token)


