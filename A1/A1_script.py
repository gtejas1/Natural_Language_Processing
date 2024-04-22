import nltk
from gensim.models import Word2Vec

# Load corpus
nltk.download('genesis')
sentences = nltk.corpus.genesis.sents()
# b = [nltk.word_tokenize(s) for s in sentences]
b = [[w.lower() for w in s]  for s in sentences]
lemmatizer = nltk.WordNetLemmatizer()
b = [[lemmatizer.lemmatize(w) for w in s]  for s in b]


print("len(sentences): " + str(len(b)))

import gensim.downloader as api
wv = api.load("word2vec-google-news-300")
print(wv.evaluate_word_pairs("C:\\Users\\Tejas\\OneDrive\\Desktop\\NLP_course\\A1\\sim_A1.txt"))
# print(wv["the"])
# print(wv.similarity("in", "the"))
print(wv.most_similar("in"))
print(wv.most_similar("the"))
print(wv.most_similar("beginning"))
print(wv.most_similar("earth"))
print(wv.most_similar("without"))

# Train two Word2Vec models with different settings
model_sg = Word2Vec(b, sg=1, vector_size=100, window=5, min_count=5, epochs=10)
model_cbow = Word2Vec(b, sg=0, vector_size=200, window=5, min_count=5, epochs=5)



model_sg.wv.save_word2vec_format("C:\\Users\\Tejas\\OneDrive\\Desktop\\NLP_course\\A1\\word2vec_sg.txt")
model_cbow.wv.save_word2vec_format("C:\\Users\\Tejas\\OneDrive\\Desktop\\NLP_course\\A1\\word2vec_cbow.txt")

print(model_sg.wv.evaluate_word_pairs("C:\\Users\\Tejas\\OneDrive\\Desktop\\NLP_course\\A1\\sim_A1.txt"))
# print(model_sg.wv["the"])
# print(model_sg.wv.similarity("in","the"))
print(model_sg.wv.most_similar("in"))
print(model_sg.wv.most_similar("the"))
print(model_sg.wv.most_similar("beginning"))
print(model_sg.wv.most_similar("earth"))
print(model_sg.wv.most_similar("without"))


print(model_cbow.wv.evaluate_word_pairs("C:\\Users\\Tejas\\OneDrive\\Desktop\\NLP_course\\A1\\sim_A1.txt"))
# print(model_cbow.wv["the"])
# print(model_cbow.wv.similarity("in","the"))
print(model_cbow.wv.most_similar("in"))
print(model_cbow.wv.most_similar("the"))
print(model_cbow.wv.most_similar("beginning"))
print(model_cbow.wv.most_similar("earth"))
print(model_cbow.wv.most_similar("without"))


import A1_helper
x_vals, y_vals, labels = A1_helper.reduce_dimensions(model_sg.wv)
A1_helper.plot_with_matplotlib(x_vals, y_vals, labels, ["in", "the", "beginning", "earth", "without", "lights", "grass", "abend","gott","das","tag","die","von","gut","so","teh","maded","et","surface","esprit"])

x_vals, y_vals, labels = A1_helper.reduce_dimensions(model_cbow.wv)
A1_helper.plot_with_matplotlib(x_vals, y_vals, labels, ["in", "the", "beginning", "earth", "without", "lights", "grass", "abend","gott","das","tag","die","von","gut","so","teh","maded","et","surface","esprit"])
print("end")