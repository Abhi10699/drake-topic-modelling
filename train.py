from pickle import load, dump
from gensim.models import LdaModel

# load corpus from pickle file

corpus = load(open("./payload/corpus.pickle","rb"))
dictionary = load(open("./payload/dictionary.pickle", "rb"))


print(f"Corpus Length: {len(corpus)}")
print(f'Number of unique tokens: {len(dictionary)}')

# HYPER PARAMETERS

num_topics = 30
chunksize = 2000
passes = 30
iterations = 400
eval_every = None 


temp = dictionary[0]
id2word = dictionary.id2token


print("[!] Training LDA Model")
model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)


# dump model 

print("[!] Dumping LDA Model")
dump(model,open("./payload/lda_model.pickle", "wb"))


top_topics = model.top_topics(corpus)
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)