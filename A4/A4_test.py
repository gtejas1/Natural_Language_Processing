from transformers import TFAutoModel
from transformers import AutoTokenizer
bert_model = TFAutoModel. from_pretrained("distilbert-base-uncased") 

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokens = tokenizer.tokenize("The dog is playing.")
tokens = tokenizer.tokenize("The dog is playing doggygame.")

token_ids = tokenizer.convert_tokens_to_ids(tokens)
tokens = ["[CLS]"] + tokens + ["[SEP]"]
token_ids = tokenizer.convert_tokens_to_ids(tokens)

t = tokenizer("The dog is playing doggygame.")

t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."])
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."],max_length=9,truncation=True)
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."],max_length=9,truncation=True,padding=True)
tokenizer.decode(t["input_ids"][1])
t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."],max_length=9,truncation=True,padding=True,return_tensors="tf") 
output = bert_model(t["input_ids"],attention_mask=t["attention_mask"])
print(output[0].shape)
print(output[0][:,0].shape)

from datasets import load_dataset
tr_dataset = load_dataset("imdb", split="train")
# tr_dataset = load_dataset("glue", "sst2")

tr_dataset = tr_dataset.shuffle(seed=0)
tr_dataset = tr_dataset[:1000]
tokenized_train = tokenizer(tr_dataset["text"] , max_length=512, truncation=True, padding=True, return_tensors="tf")
ts_dataset = load_dataset("imdb", split="test")


ts_dataset = ts_dataset.shuffle(seed=0)
ts_dataset = ts_dataset[:1000]
tokenized_test = tokenizer(ts_dataset["text"] , max_length=512, truncation=True, padding=True, return_tensors="tf")

from tensorflow.keras.utils import to_categorical
train_y = to_categorical(tr_dataset["label"])

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False
maxlen = 512  	
token_ids = Input(shape=(maxlen,), dtype=tf.int32, name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32, name="attention_masks")
 
bert_output = bert_model(token_ids,attention_mask=attention_masks)

dense_layer = Dense(64,activation="relu")(bert_output[0][:,0])

output = Dense(2,activation="softmax")(dense_layer)

model = Model(inputs=[token_ids,attention_masks],outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],train_y, batch_size=25, epochs=1)
test_y = to_categorical(ts_dataset["label"])
score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]], test_y, verbose=0)
print("Accuracy on test data:", score[1])

a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]]) 

print(a.shape)
print(a[0])
print(a[1])

t = tokenizer(["The dog is playing doggygame.", "The cat is sleeping."], max_length=9, truncation=True, padding=True)
output = bert_model(t["input_ids"],attention_mask=t["attention_mask"])


import numpy as np
from math import sqrt
def cosine_similarity(a, b) :
	return np.dot(a,b)/(sqrt(np.dot(a,a))*sqrt(np.dot(b,b)) )

print(cosine_similarity(output[0][0][2], output[0][1][2]))