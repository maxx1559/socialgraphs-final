import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nltk.tokenize import word_tokenize  #type:ignore
from collections import defaultdict

#Define categories
subjects = {
    'scientific': ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
    'political': ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc'],
    'technical': ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                 'comp.sys.mac.hardware', 'comp.windows.x'],
    'forsale': ['misc.forsale'],
    'religious': ['soc.religion.christian', 'talk.religion.misc', 'alt.atheism'],
    'recreational': ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
}

#Create a mapping from original subject to new merged subject
subject_mapping = {}
for new_subject, old_subjects in subjects.items():
    for subject in old_subjects:
        subject_mapping[subject] = new_subject

#The wordlist can be found at https://doi.org/10.1371/journal.pone.0026752.s001
print("Processing data... Estimated time: 1-2 minutes.")
CSV_NAME = "data.csv"
WORDLIST_NAME = os.path.join("Jonatan", "LabMT_wordlist.txt")
newsgroup_df = pd.read_csv(CSV_NAME)
wordlist_df = pd.read_csv(WORDLIST_NAME, sep='\t', na_values='--')

#Map original labels to new categories
newsgroup_df['label'] = newsgroup_df['label'].map(subject_mapping)
sentiment_dict = dict(zip(wordlist_df['word'], wordlist_df['happiness_average']))

def calculate_sentiment(text, sentiment_dct):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        scores = [sentiment_dct[word] for word in tokens if word in sentiment_dct]
        return np.mean(scores) if scores else None
    return None

newsgroup_df['sentiment'] = newsgroup_df['text'].apply(
    lambda x: calculate_sentiment(x, sentiment_dict)
)

#Group by label and calculate statistics
group_stats = defaultdict(dict) #type:ignore
for label in newsgroup_df['label'].unique():
    group_sentiments = newsgroup_df[newsgroup_df['label'] == label]['sentiment']
    group_stats[label] = {
        'mean': np.mean(group_sentiments),
        'std': np.std(group_sentiments),
        'size': len(group_sentiments)
    }
#Sort groups by mean sentiment
sorted_grps = sorted(group_stats.items(),
                      key=lambda x: x[1]['mean'],
                      reverse=True)
plt.figure(figsize=(3, 2.5))
grps = [g[0] for g in sorted_grps]
sentiments = [g[1]['mean'] for g in sorted_grps]
errors = [g[1]['std'] for g in sorted_grps]

plt.errorbar(range(len(grps)), sentiments, yerr=errors, fmt='o')
plt.xticks(range(len(grps)), grps, rotation=45, ha='right')
plt.ylabel('Average Score')
plt.title('Newsgroup Sentiment by Subject')
plt.tight_layout()
print("\nDone: Close the plot window to continue...")
plt.show()

print("\nNewsgroup Sentiment Statistics:")
for group, stats in sorted_grps:
    print(f"{group}: {stats['mean']:.3f} (±{stats['std']:.3f}), size: {stats['size']}")
all_sents = newsgroup_df['sentiment'].dropna()
print("\nOverall Statistics:")
print(f"mean: {np.mean(all_sents):.3f}")
print(f"median: {np.median(all_sents):.3f}")
print(f"std: {np.std(all_sents):.3f}")
print(f"25th percentile: {np.percentile(all_sents, 25):.3f}")
print(f"75th percentile: {np.percentile(all_sents, 75):.3f}")