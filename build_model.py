import json
import time, re, numpy as np, langid
from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.linear_model import SGDClassifier

start_time = time.clock()
# read the training file first
x = []
with open(r'../../train_X_languages_homework.json.txt') as f:
    objects = f.readlines()
    for j_obj in objects:
        feature = json.loads(j_obj, encoding='utf-8')
        feature = feature['text']
        feature_clean = re.sub(r"[-()\"#•/@\[\];:<>{}`%+=~|（.!?,·\d，]", "", feature)
        feature_clean = re.sub(r"\s+", " ", feature_clean)
        feature_clean = feature_clean.strip().lower()
        x.append(feature_clean)

# reading labels for training data
y = []
with open(r'../../train_y_languages_homework.json.txt') as f:
    objects = f.readlines()
    for j_obj in objects:
        feature = json.loads(j_obj, encoding='utf-8')
        y.append(feature['classification'])

label_names, counts = np.unique(y, return_counts=True)


def under_sample(texts, label):
    # all records for a particular language are undersampled to a default of 500 records
    len_dict = {}
    sentences = []
    langs = []
    to_clean_langs_label = ['ca', 'cs', 'da', 'de', 'en', 'eo', 'es', 'et', 'eu', 'fi', 'fr', 'gl', 'hr', 'hu', 'id',
                            'it', 'la', 'lt', 'ms', 'nl', 'nn', 'no', 'pl', 'pt', 'ro', 'ru', 'sk', 'sl', 'sr', 'sv',
                            'tr', 'uk', 'vi', 'zh']
    for lang in label_names:
        len_dict[lang] = 0
    for lang, text in zip(label, texts):
        if len_dict[lang] <= 1000 and (10 < len(text) <= 140):
            if lang in to_clean_langs_label:
                if langid.classify(text)[0] != lang: continue
            len_dict[lang] += 1
            langs.append(lang)
            sentences.append(text)
    return langs, sentences


y_resampled, x_resampled = under_sample(x, y)

with open('test_training.txt', 'w+', encoding='utf-8') as f:
    for i in range(len(y_resampled)):
        if y_resampled[i] == 'sh':
            f.write(y_resampled[i] + " " + x_resampled[i] + "\n")

x_train, x_val, y_train, y_val = model_selection.train_test_split(x_resampled, y_resampled, test_size=0.2,
                                                                  random_state=42)

vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 6), analyzer='char')

pipe = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', SGDClassifier(n_jobs=-1))
])

print('model compilation done, begin fitting')

scores = model_selection.cross_val_score(pipe, x_resampled, y_resampled, cv=5)

print("Score are ", scores.mean(), scores.std()*2)

# pipe.fit(x_train, y_train)
#
# y_predicted = pipe.predict(x_val)

dump(pipe, 'model.bin')
#
# cm = metrics.confusion_matrix(y_val, y_predicted)
# percentage_matrix = 100 * cm / cm.sum(axis=1).astype(float)
#
# plt.figure(figsize=(20, 15))
# ax = plt.subplot(111)
# sns.heatmap(percentage_matrix, annot=True, fmt='0.0f', xticklabels=label_names, yticklabels=label_names, ax=ax)
# plt.title('Confusion Matrix for Languages')
# fig1 = plt.gcf()
# fig1.savefig(r'plot_new.jpg', dpi=100)
# print(metrics.accuracy_score(y_val, y_predicted))
# print(metrics.classification_report(y_val, y_predicted, target_names=label_names))
# print(time.clock() - start_time, 'seconds')
