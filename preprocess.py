import json
import re
import spacy
import pickle


from tqdm import tqdm
from gensim.models import Phrases
from gensim.corpora import Dictionary



# initialize spacy

nlp = spacy.load("en_core_web_sm")

# read json file

def load_data(file_path):
  with open(file_path,"r") as f:
    file_json = json.load(f)
    f.close()
  return  file_json

drake_data = load_data("./data/drake_data.json")
docs = []

def remove_verse_label(text):
  regex = re.compile("(\[.*])")
  text = re.sub(regex,"", text)
  text = text.strip()
  return text


def preprocess_song(song_lyrics):

  # lowercase 

  song_lyrics = song_lyrics.lower()

  # remove unnecessary phrases
  song_lyrics = remove_verse_label(song_lyrics)

  # trim new lines

  song_lyrics = song_lyrics.replace("\n"," ").strip()
  song_lyrics = song_lyrics.replace("  "," ").strip()

  # Tokenize the sentences

  tokenized = []

  for token in nlp(song_lyrics):
    if not (token.is_stop or token.is_punct or token.is_space or token.is_punct or token.is_quote or len(token.text) <= 3):
      tokenized.append(token.lemma_)


  return tokenized



print("[!] Cleaning Lyrics..")

for song in tqdm(drake_data):
  if song['lyrics'] is not None:
    song_lyrics = song['lyrics']
    cleaned_lyrics = preprocess_song(song_lyrics)
    docs.append(cleaned_lyrics)


bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)


print("[!] Building Dictionary")
# build word dictionary
dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=20, no_above=0.5)


print("[!] Building bag of words")
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('\tNumber of unique tokens: %d' % len(dictionary))
print('\tNumber of documents: %d' % len(corpus))



# dump corpus to a pickle file
print("\n\n[!] Dumping to pickle file")

pickle.dump(corpus, open("./payload/corpus.pickle", "wb"))
pickle.dump(dictionary, open("./payload/dictionary.pickle", "wb"))

