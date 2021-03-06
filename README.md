# Language_identification
Identifying  53 languages using SVM classification over tf-idf features
he training module is dependent on below packages:

	• JSON, TIME, RE,  NUMPY, LANGID
	• Punctuations (String), SKLEARN (feature extraction, model selection, metrics, SGDClassifier)
	• MATPLOTLIB, seaborn

Feature engineering:

The only column provided was the text with a sequnce of symbols/characters which essentially is not much useful for any classification algorithm. Hence, the text was vectorized at a character level of range 1 to 6. 
This allowed any word to be broken into an ngram of 1 to 6 length and each of these ngram were then converted to 
Tf-idf matrix representation which was used as features further on for the classification algorithm to fit on.

Before vectorization , below operation were performed to clean the text:

	• Removed any special characters from the text using the regex library in python
	• Stripped away any extra white spaces in between words and at the start/end of the text
	• Removed all punctuation symbols
	• Discarded any texts from the training corpus less than 10 and more than 140 characters to keep all sentences length balanced and meaningful at the same time
	• Removed any multilingual text from training corpus by only allowing alphabets of common languages. For e.g for English language only alphabets [a-z] were allowed and similarly for other common languages.
	• Remove any numerical data [0-9] from the training corpus as it will of no use since every language will have the same representation.
	

How the model was chosen:

The model chosen was SVM (Support vector model) with SGD (stochastic gradient descent) optimization method. SVM is chosen over logistic regression because of it's ability to use a kernel to transform the data and then find an optimal boundary between the possible labels.

Other important contributor to the choice of SVM was the # of features, since we're essentially using tf-idf matrix representation of ngrams as features, we end up having huge number of features and hence SVM was a natural choice.


Other options to explore:

If given more time I'd invest more time to clean the data more comprehensively so that no mislabelled word can creep in.
