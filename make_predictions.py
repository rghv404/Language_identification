import re, json, time
from string import punctuation
from joblib import load

start_time = time.clock()
x_test = []

with open(r'..\test_X_languages_homework.json.txt') as f:
    objects = f.readlines()
    for j_obj in objects:
        feature = json.loads(j_obj, encoding='utf-8')
        feature = feature['text']
        feature_clean = re.sub(r"[-()\"#•/@\[\];:<>{}`%+=~|（.!?,·\d。，]", "", feature)
        feature_clean = re.sub(r"\s+", " ", feature_clean)
        feature_clean = feature_clean.strip()
        punc = list(punctuation)
        temp = filter(lambda x: x not in punc + [',', "’", '!', ':', "–"], feature_clean)
        clean_text = "".join(ch for ch in list(temp))
        x_test.append(clean_text.lower())

# load the mode
pipe = load(r'model.bin', 'r')

# x_test = vectorizer.transform(x_test)
y_predicted = pipe.predict(x_test)

# write the predicted values to file
with open(r'predictions.txt', 'w+', encoding='utf-8') as f:
    for y in y_predicted:
        # create json object
        predictions = dict()
        predictions['classification'] = y
        json_obj = json.dumps(predictions)
        f.write(json_obj + "\n")

print(time.clock() - start_time, 'seconds')