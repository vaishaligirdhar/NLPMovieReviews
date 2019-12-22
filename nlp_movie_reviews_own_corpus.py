import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import PlaintextCorpusReader
import nltk.data

# load corpus
corpus_root = './review_polarity/txt_sentoken'
wordlists = nltk.corpus.reader.PlaintextCorpusReader(corpus_root, '.*')

# define word_feats which is a simple dictionary mapping
# feature names to feature values
def word_feats(words):
    return dict([(word, True) for word in words])

# define location of negative and positive data samples
negids = ['neg']
posids = ['pos']

# map feature names to feature values
negfeats = [(word_feats(wordlists.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(wordlists.words(fileids=[f])), 'pos') for f in posids]

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