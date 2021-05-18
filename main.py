from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel 
import pandas as pd 
import numpy as np 
import requests
import re 
import hdbscan
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

app = FastAPI()

# class Item(BaseModel):
#     name: str 
#     price: float 
#     is_offer: Optional[bool] = None
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/urls/{urls}")
def read_item(urls: str, q: Optional[str] = None):


    names = ['Salsa dancing', 'Sensory deprivation', 'Marathon', 'Software','Financial institution', 'Random forests','Linear regression','Football', 'Natural language Processing', 'Handstands', 'Yoga', 'Cluster analysis', 'Water polo', 'Teller','Supervised Learning','Cyber Security','Data Science','Artificial intelligence','European Central Bank','Financial technology','International Monetary Fund','Basketball','Swimming', 'Soccer']
    my_url = []
    title=[]
    # for name in names:
    #     site = 'en.wikipedia.org/wiki/' + name
    #     print(site)
    #     my_url.append(site)
    urls = urls.split(", ")

    for url in urls:
        site = 'en.wikipedia.org/wiki/' + url
        print(site)
        my_url.append(site)
    
    def create_input_df(my_url):
        print('==================================> inside create')
        for u in my_url:
            URL = 'http://' + u
            session = requests.Session()
            retry = Retry(connect=3, backoff_factor=0.5)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            # session.get(url)
            response = session.get(URL)
            html_doc = response.content
            soup = BeautifulSoup(html_doc, 'html.parser')
            text = soup.find('div', {'id': 'mw-content-text'}).get_text()
            text = text[:500]
            d['url'].append(URL)
            d['title'].append(u[22:])
            d['text'].append(text)
        df = pd.DataFrame(d) 
        return df 

    d = {'url': [], 'title': [],'text': []}
    df = create_input_df(my_url)
    

    def review_to_words(raw_review):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and 
        # the output is a single string (a preprocessed movie review)
        
        # 1. Remove HTML
        review_text = BeautifulSoup(raw_review, features='html.parser').get_text() 
        #
        # 2. Remove line-breaks (\n)
        line_breaks = review_text.strip().replace('\n', '')
        
        # 2. Remove non-letters        
        letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()                             
        #
        # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))                  
        # 
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]   
        #
        # 6. Join the words back into one string separated by space, and return the result.
        return( " ".join( meaningful_words ))

    df.text = df.text.apply(review_to_words)
    
    #tokenize and tag the card text
    card_docs = [TaggedDocument(doc.split(' '), [i]) 
                for i, doc in enumerate(df.text)]

    #instantiate model
    model = Doc2Vec(vector_size=64, window=2, min_count=1, workers=8, epochs = 40)
    #build vocab
    model.build_vocab(card_docs)
    #train model
    model.train(card_docs, total_examples=model.corpus_count
                , epochs=model.epochs)
    
    #generate vectors
    card2vec = [model.infer_vector((df['text'][i].split(' '))) 
                for i in range(0,len(df['text']))]

    #Create a list of lists
    dtv= np.array(card2vec, dtype='object').tolist()
    #set list to dataframe column
    df['card2vec'] = dtv
     

    # umap to simplify embeddings
    umap_embeddings = umap.UMAP(n_neighbors=2, 
                                n_components=2, 
                                metric='cosine').fit_transform(dtv)
    
    # hdbscan to cluster embeddings
    cluster = hdbscan.HDBSCAN(min_cluster_size=2,
                            metric='euclidean',                      
                            cluster_selection_method='eom').fit(umap_embeddings)
    
    # ########################
    # # VISUALIZATION
    # # Prepare data
    # umap_data = umap.UMAP(n_neighbors=2, n_components=2, min_dist=0.0, metric='cosine').fit_transform(dtv)
    # result = pd.DataFrame(umap_data, columns=['x', 'y'])
    # result['labels'] = cluster.labels_

    # # Visualize clusters
    # fig, ax = plt.subplots(figsize=(20, 10))
    # outliers = result.loc[result.labels == -1, :]
    # clustered = result.loc[result.labels != -1, :]
    # plt.scatter(outliers.x, outliers.y, color='#BDADBD', s=10)
    # plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=10, cmap='hsv_r')
    # plt.colorbar()
    # plt.show();
    # #########################

    # # new df with topic clusters (numbers)
    docs_df = pd.DataFrame(df, columns=['title',"text"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'text': ' '.join})

    # #tfidf but for entire documents in the same cluster number
    def c_tf_idf(documents, m, ngram_range=(1, 1)):
        count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
        t = count.transform(documents).toarray()
        w = t.sum(axis=1)
        tf = np.divide(t.T, w)
        sum_t = t.sum(axis=0)
        idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
        tf_idf = np.multiply(tf, idf)

        return tf_idf, count
    
    tf_idf, count = c_tf_idf(docs_per_topic.text.values, m=len(df))

    # # want to get the top words per topic 
    def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
        words = count.get_feature_names()
        labels = list(docs_per_topic.Topic)
        tf_idf_transposed = tf_idf.T
        indices = tf_idf_transposed.argsort()[:, -n:]
        top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
        return top_n_words

    def extract_topic_sizes(df):
        topic_sizes = (df.groupby(['Topic'])
                        .text
                        .count()
                        .reset_index()
                        .rename({"Topic": "Topic", "text": "Size"}, axis='columns')
                        .sort_values("Size", ascending=False))
        return topic_sizes

    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)

    # gimme gimme the labels 
    docs_df['top_name'] = docs_df.Topic.map(lambda x: top_n_words[x][0][0])

    docs_df
    return {"Categories": docs_df[['title', 'top_name']]} #, "q": q}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Optional[str] = None):
#     return {"item_id": item_id, "q": q}

# @app.put(".items/{item_id")
# def update_item(item_id: int, item: Item):
#     return {"item_price": item.price, "item_id": item_id}