# Latent Dirichlet Allocation (LDA)
import pandas as pd
from datetime import datetime
from gensim.corpora import Dictionary  #type:ignore
from gensim.models import LdaModel  #type:ignore
from nltk.tokenize import word_tokenize  #type:ignore
from nltk.corpus import stopwords  #type:ignore
from nltk.stem import WordNetLemmatizer  #type:ignore
import nltk  #type:ignore
import os
import pickle
import string
import time

nltk.download('punkt_tab')

OUTPUT_DIR = 'models_lda'
LDA_NUM_TOPICS = 6
LDA_NUM_PASSES = 100
FILTER_BELOW = 2  #ignore rare words that exist in fewer than 2 documents
FILTER_ABOVE = 0.7  #ignore common words that occour in more than 90% of the documents

def run_lda_analysis(csv_path, num_topics=LDA_NUM_TOPICS, num_passes=LDA_NUM_PASSES):
    """
    Run LDA topic modeling on text data from a CSV file.
    num_topics (int): topics for LDA model (default: 5)
    num_passes (int): passes for LDA training (default: 30)
    """
    print("Starting the LDA process...")
    start_time = time.time()

    # Automatically download required NLTK data
    print("\nChecking and downloading required NLTK data...")
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for item in required_nltk_data:
        try:
            nltk.data.find(f'tokenizers/{item}' if item == 'punkt'
                          else f'corpora/{item}' if item in ['stopwords', 'wordnet']
                          else f'taggers/{item}')
        except LookupError:
            print(f"Downloading {item}...")
            nltk.download(item)

    #part1
    print("\n1. Loading CSV file...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} documents")

    #part2
    print("\n2. Initializing NLTK components...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    print("   Initialized lemmatizer and stopwords")

    def preprocess(text):
        text = str(text).lower()  #convert to string
        text = text.translate(str.maketrans('', '', string.punctuation))  #remove punctuation
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]  #remove stopwords and lemmatize
        return tokens

    #part3
    print("\n3. Preprocessing texts...")
    texts = []
    for i, doc in enumerate(df['text']):
        if i % 1000 == 0:  #print progress every 1000doc
            print(f"   Processing document {i}/{len(df)}")
        texts.append(preprocess(doc))
    print("   Preprocessing complete")

    #part4
    print("\n4. Creating dictionary...")
    dictionary = Dictionary(texts)
    print(f"   Initial dictionary size: {len(dictionary)}")

    #part5
    print("\n5. Filtering dictionary...")
    dictionary.filter_extremes(no_below=FILTER_BELOW, no_above=FILTER_ABOVE)  #filter out extreme frequencies
    print(f"   Dictionary size after filtering: {len(dictionary)}")

    #part6
    print("\n6. Creating corpus...")
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f"   Corpus size: {len(corpus)}")

    #part7
    print("\n7. Training LDA model...")
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=num_passes)
    print("   LDA model training complete")

    #part8
    print("\n8. Final Topics:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"\nTopic {idx}:")
        print(topic)

    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time):.2f} seconds")
    print("\nAdditional Statistics:")  #print more statistics yay
    print(f"Number of documents processed: {len(df)}")
    print(f"Final vocabulary size: {len(dictionary)}")
    print(f"Average tokens per document: {sum(len(text) for text in texts)/len(texts):.1f}")
    return lda_model, dictionary, corpus, texts

def save_model_results(lda_model, dictionary, corpus, texts, output_dir=OUTPUT_DIR):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  #generate timestamp for unique filenames each time code is run
    model_path = os.path.join(output_dir, f'lda_model_{timestamp}')
    lda_model.save(model_path)  #save model
    dict_path = os.path.join(output_dir, f'dictionary_{timestamp}.dict')
    dictionary.save(dict_path)  #save dictionary

    corpus_path = os.path.join(output_dir, f'corpus_{timestamp}.pkl')  #save corpus and texts using pickle
    texts_path = os.path.join(output_dir, f'texts_{timestamp}.pkl')  #save corpus and texts using pickle
    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus, f)
    with open(texts_path, 'wb') as f:
        pickle.dump(texts, f)
    print(f"\nAll data saved in {output_dir}")
    return timestamp

if __name__ == "__main__":
    # Configuration
    CSV_PATH = 'processed_data.csv'
    try:
        _lda_model, _dictionary, _corpus, _texts = run_lda_analysis(
            csv_path=CSV_PATH,
            num_topics=LDA_NUM_TOPICS,
            num_passes=LDA_NUM_PASSES
        )
        save_model_results(_lda_model, _dictionary, _corpus, _texts)
    except Exception as e:
        print(f"An error occurred: {str(e)}")