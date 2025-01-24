import pandas as pd

#Loading dataset
languages = pd.read_csv('Language Detection.csv')
languages = languages
print(languages)

text = languages['Text']
lang = languages['Language']
# print(lang)
# print(text

import re
data_list = []
for texts in text:
        #print(texts)
        texts = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', texts)
        texts = re.sub(r'[[]]', ' ', texts)
        texts = texts.lower()
        data_list.append(texts)

print(data_list)

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
x= cv.fit_transform(data_list).toarray()
print(x)

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
y = tf.fit_transform(data_list).toarray()
print(y)
X=y

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(lang)
print(X)
print(y)

from sklearn.naive_bayes import MultinomialNB
langIden_model = MultinomialNB().fit(X_train, y_train)
y_pred=langIden_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Classification report (includes precision, recall, f1 for each class)
class_report = classification_report(y_test, y_pred)

# Printing results
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-Score: {f1}")
# print("Confusion Matrix:")
# print(conf_matrix)
# print("\nClassification Report:")
# print(class_report)

import seaborn as sns
import matplotlib.pyplot as plt

# Visualize confusion matrix with heatmap
plt.figure(figsize=(10,8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

def predict(text):
    x = cv.transform([text]).toarray()
    lang = langIden_model.predict(x)
    lang = le.inverse_transform(lang)
    print("The language is in",lang[0])
text = "Wie geht es dir"
predict(text)