import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

# define word_feats which is a simple dictionary mapping
# feature names to feature values
def word_feats(words):
    return dict([(word, True) for word in words])

# define location of negative and positive data samples
negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

# map feature names to feature values
negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

# slice dataset into training set and test set
negcutoff = len(negfeats)*4//5
poscutoff = len(posfeats)*4//5
 
trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

# train using NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(trainfeats)

# show accuracy of text classification
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))

# show the most valuable/useful features
classifier.show_most_informative_features()