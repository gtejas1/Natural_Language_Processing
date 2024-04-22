from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from math import sqrt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


train_dataset = load_dataset("dair-ai/emotion", split="train")
test_dataset = load_dataset("dair-ai/emotion",split="test")

limited_train_data = train_dataset[:1000] 
limited_test_data = test_dataset[:1000]  

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_train = tokenizer(limited_train_data["text"][:1000], padding='max_length', max_length=512, truncation=True, return_tensors="tf")
tokenized_test = tokenizer(limited_test_data["text"][:1000], padding='max_length', max_length=512, truncation=True, return_tensors="tf")

train_y = to_categorical(limited_train_data["label"])
test_y = to_categorical(limited_test_data["label"])

bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")
bert_model.trainable = False

maxlen = 512
token_ids = Input(shape=(maxlen,), dtype=tf.int32,name="token_ids")
attention_masks = Input(shape=(maxlen,), dtype=tf.int32, name="attention_masks")
bert_output = bert_model(token_ids,attention_mask=attention_masks)
dense_layer = Dense(64,activation="relu")(bert_output[0][:,0]) 


output = Dense(6,activation="softmax")(dense_layer)


model = Model(inputs=[token_ids,attention_masks],outputs=output)
print( model.summary())
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) 
model.fit([tokenized_train["input_ids"],tokenized_train["attention_mask"]],train_y, batch_size=20, epochs=1)

score = model.evaluate([tokenized_test["input_ids"],tokenized_test["attention_mask"]], test_y, verbose=0)

print("\nAccuracy on test data:", score[1])
print("Loss on test data:", score[0])
print('\n')

a = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
print("a.shape")
print(a.shape)
print("a[0]")
print(a[0])

predicted_test_probabilities = model.predict([tokenized_test["input_ids"], tokenized_test["attention_mask"]])
predicted_test_labels = np.argmax(predicted_test_probabilities, axis=1)

actual_test_labels = np.argmax(test_y, axis=1)


cp = []
ip = []

for i in range(len(predicted_test_labels)):
    if predicted_test_labels[i] == actual_test_labels[i]:
        cp.append((limited_test_data["text"][i], predicted_test_labels[i], actual_test_labels[i]))
    else:
        ip.append((limited_test_data["text"][i], predicted_test_labels[i], actual_test_labels[i]))

print("Correct Pred:")
for text, predicted, actual in cp[10:21]:
    print(f"Text: {text}, Predicted: {predicted}, Actual: {actual}")

print("\nIncorrect Pred:")
for text, predicted, actual in ip[10:21]:
    print(f"Text: {text}, Predicted: {predicted}, Actual: {actual}")


def cosine_similarity(a, b):
    return np.dot(a, b) / (sqrt(np.dot(a, a)) * sqrt(np.dot(b, b)))

sentence_pairs = [
    ("The chef prepared a delicious meal in the kitchen.", "The aroma of freshly baked bread filled the air."),
    ("The students eagerly awaited the exam results.", "The teacher explained the new lesson with enthusiasm."),
    ("The athlete sprinted across the finish line.", "The crowd cheered loudly for the winning team."),
    ("The scientist conducted experiments in the laboratory.", "The microscope revealed intricate details of the specimen."),
    ("The sun set behind the mountains in a spectacular display.", "The stars began to twinkle in the night sky.")
]

for sentence1, sentence2 in sentence_pairs:
    tokens = tokenizer([sentence1,sentence2], padding=True, truncation=True, return_tensors="tf")

   
    output = bert_model(tokens["input_ids"], attention_mask=tokens["attention_mask"])

    similartyscore = cosine_similarity(output[0][0][2], output[0][1][2])
    print(f"Cosine similarity for  '{sentence1}' and '{sentence2}' is : {similartyscore}")
