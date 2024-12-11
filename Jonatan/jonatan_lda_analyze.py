# Latent Dirichlet Allocation (LDA)
import matplotlib.pyplot as plt
import seaborn as sns  #type:ignore
import numpy as np
import os
import pickle
import pandas as pd
from gensim.models import LdaModel  #type:ignore
from gensim.corpora import Dictionary  #type:ignore
from datetime import datetime

OUTPUT_DIR = 'models_lda'

def analyze_lda_results(lda_model, corpus):
    print("\n=== LDA Analysis Results ===")

    #top words
    print("\n1. Topics and their top words:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"\nTopic {idx}:")
        # Convert the topic string to a more readable format
        words = [(word.split('*')[1].strip().replace('"', ''), float(word.split('*')[0]))
                for word in topic.split(' + ')]
        for word, prob in words:
            print(f"  - {word}: {prob:.4f}")

    #document-topic distribution for first few documents
    print("\n2. Topic distribution for first 5 documents:")
    for i, doc_topics in enumerate(lda_model[corpus[:5]]):
        print(f"\nDocument {i}:")
        # Sort topics by probability for this document
        sorted_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
        for topic_id, prob in sorted_topics:
            print(f"  Topic {topic_id}: {prob:.4f}")

    #calculate and show dominant topic for every doc
    print("\n3. Document dominant topics summary:")
    doc_topics = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
    topic_counts = pd.Series(doc_topics).value_counts()
    print("\nNumber of documents per dominant topic:")
    for topic_id, count in topic_counts.items():
        print(f"Topic {topic_id}: {count} documents")

    #show coherence score (if available)
    try:
        coherence = lda_model.coherence
        print(f"\n4. Model coherence score: {coherence}")
    except:  #pylint: disable=W0702
        print("\n4. Coherence score not available")

def visualize_lda_results(lda_model, corpus):
    #not in use for now
    #topic distribution across documents
    doc_topics = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
    plt.figure(figsize=(10, 6))
    sns.countplot(x=doc_topics)
    plt.title('Distribution of Dominant Topics Across Documents')
    plt.xlabel('Topic ID')
    plt.ylabel('Number of Documents')
    plt.show()


def analyze_specific_documents(lda_model, corpus):
    # Get topics for a specific document
    doc_id = 0  # first document
    print(f"\nTopics for document {doc_id}:")
    for topic_id, prob in lda_model[corpus[doc_id]]:
        print(f"Topic {topic_id}: {prob}")

    # Get the most probable words for a specific topic
    topic_id = 0  # first topic
    print(f"\nTop words for topic {topic_id}:")
    words = lda_model.show_topic(topic_id, topn=10)
    for word, prob in words:
        print(f"{word}: {prob:.4f}")

def load_and_analyze_results(model_path, dict_path, corpus_path, texts_path):
    #load the output of the training script (you must run train script first)
    loaded_model = LdaModel.load(model_path)
    loaded_dict = Dictionary.load(dict_path)
    with open(corpus_path, 'rb') as f:
        loaded_corpus = pickle.load(f)
    with open(texts_path, 'rb') as f:
        loaded_texts = pickle.load(f)
    return loaded_model, loaded_dict, loaded_corpus, loaded_texts

if __name__ == "__main__":
    #find the most recent model files in the output directory
    #then get the timestamp from the most recent model file
    #this is such that we can have multiple models, but the code automatically finds the newest
    #probably overkill and not necessary :C
    _files = os.listdir(OUTPUT_DIR)
    _model_files = [_f for _f in _files if _f.startswith('lda_model_')]
    if not _model_files:
        print("No model files found in output directory")
        exit()
    _latest_model = max(_model_files)  #get the base model name without the extension
    _timestamp = _latest_model.split('.')[0]  #remove file extension

    _model_path = os.path.join(OUTPUT_DIR, _timestamp)  #construct paths for all files
    _dict_path = os.path.join(OUTPUT_DIR, f'{_timestamp.replace("lda_model", "dictionary")}.dict')
    _corpus_path = os.path.join(OUTPUT_DIR, f'{_timestamp.replace("lda_model", "corpus")}.pkl')
    _texts_path = os.path.join(OUTPUT_DIR, f'{_timestamp.replace("lda_model", "texts")}.pkl')

    print(f"Loading files with timestamp: {_timestamp}")
    print(f"Model path: {_model_path}")
    print(f"Dictionary path: {_dict_path}")
    print(f"Corpus path: {_corpus_path}")
    print(f"Texts path: {_texts_path}")

    try:
        _loaded_model, _loaded_dict, _loaded_corpus, _loaded_texts = load_and_analyze_results(
            _model_path, _dict_path, _corpus_path, _texts_path
        )  #load all data (run the train script first)
        analyze_lda_results(_loaded_model, _loaded_corpus)

    except Exception as e:
        print(f"An error occurred: {str(e)}")