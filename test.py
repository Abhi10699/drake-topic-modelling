from pickle import load

lda_model = load(open("./payload/lda_model.pickle", "rb"))
# print(dir(lda_model))


for topic in lda_model.show_topics(): 
  print(topic)