from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords

s=stopwords.words('english')

tags = {}
frequent_tags = []
count = 0
ftag = open('Tags.txt', 'rt', encoding="utf-8")
fclean = open('cleanedBody.txt', 'rt', encoding="utf-8")
tag_rows = ftag.read().split('\n')[:150000]
# X-> cleaned body
X = fclean.read().split('\n')[:150000]
# Y-> most repeated tag for that title, so basically most probable tag for the given title
Y = [[] for i in range(len(X))]
classifier = Pipeline([('vectorizer', CountVectorizer(analyzer='word',stop_words = s, lowercase = True, ngram_range=(1, 2))),
('tfidf', TfidfTransformer()),
('clf', OneVsRestClassifier(LinearSVC(C=1.5, penalty='l2', dual=True), n_jobs=-1))])
# Frequency of each tag
for line in tag_rows:
    for tag in line.split():
        if tag in tags:
            tags[tag] += 1
        else:
            tags[tag] = 1
    count = 0
#Any tag that appears more than 4000 times in the dataset is considered frequent.
# reverse sorted the dictionary so that frequent tags are arranged from higher frequency to lower frequency.
for tag in sorted(tags, key=lambda tag: tags[tag], reverse=True):
    if tags[tag] > 4000:
        count += 1
        frequent_tags.append(tag)
    else:
        break
#From the tags of each record, only those that occur in the frquent_tags list will be assigned as tags to that record. 
#Example:# X-> check upload file imag without mime type bis way check upload file imag apart check file extens use php
         # Y-> php
for x, tag in enumerate(frequent_tags):
    i = 0
    for row in tag_rows:
        if tag in row.split():
            Y[i].append(tag)
        i = i + 1

print("Tag_rows:", tag_rows[:10])
print("Frequency:", frequent_tags[:10])
print("X:", X[:10])
print("Y", Y[:10])
multibin = MultiLabelBinarizer()
Y = multibin.fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=None)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred)*100)
print("Precision:",precision_score(y_test, y_pred, average='micro') * 100)
print("Recall:",recall_score(y_test, y_pred, average='micro') * 100)
print("F1Score:",f1_score(y_test, y_pred, average='micro') * 100)
print("Hamming Loss:",hamming_loss(y_test, y_pred)*100)
