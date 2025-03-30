import itertools
from lxml import etree
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from enum import Enum


# Class representing a single verse
class Verse:
    # Create object of Verse
    def __init__(self, verse) -> None:
        self.verse = verse

    # Get original verse
    def getOriginalVerse(self):
        return self.verse

    # Get normalized verse (apostrophes removed as we follow a rather phonetic understanding of the words)
    def getNormalizedVerse(self):
        # Remove apostrophes
        return self.verse.replace("\' ", "").replace("’ ", "").replace(" \'", "").replace(" ’", "").replace("\'", "").replace("’", "")

    # Get sillabified verse using spacy
    def getSillabizedVerse(self, nlp):
        doc = nlp(self.getNormalizedVerse())
        newPhraseList = [
            token._.syllables for token in doc if token._.syllables != None]
        newPhraseList = list(itertools.chain.from_iterable(newPhraseList))
        newPhrase = ' '.join(newPhraseList)
        return newPhrase


# Class representing a whole text (TEI XML)
class Text:
    # Create object of Text
    def __init__(self, path) -> None:
        # Get verses from TEI XML file (using lines, i.e. l-tag)
        xmlTree = etree.parse(path)
        lines = xmlTree.xpath('//l')
        self.verses = [Verse(''.join(line.itertext()))
                       for line in lines if line.text != None]

    # Get all verses in sillabified form
    def getSillabifiedVerses(self, nlp):
        return [sillVerse.getSillabizedVerse(nlp) for sillVerse in self.verses if sillVerse != None]

    # Get random verses from text
    def getRandomVerses(self, numberOfVerses):
        return random.sample(self.verses, numberOfVerses)


# Class representing a whole text (TXT file)
class TxtText:
    # Create object of Text
    def __init__(self, path) -> None:
        # Get verses from TXT file (lines as verses)
        f = open(path, mode="r", encoding="utf-8")
        lines = f.readlines()
        f.close()
        self.verses = [Verse(line)
                       for line in lines if line != None and line != ""]

    # Get all verses in sillabified form
    def getSillabifiedVerses(self, nlp):
        return [sillVerse.getSillabizedVerse(nlp) for sillVerse in self.verses if sillVerse != None]

    # Get random verses from text
    def getRandomVerses(self, numberOfVerses):
        return random.sample(self.verses, numberOfVerses)


# Class representing a whole text (corpus)
class TextCorpus:
    # Create object of Text
    def __init__(self, textData, labels, trainratio=.8, valratio=.2, shuffles=5) -> None:
        # Set textData and labels
        self.textData = textData
        self.labels = labels

        # Handle ratios
        self.trainratio = trainratio
        self.valratio = valratio

        dataList = [self.textData, self.labels]
        self.completeCorpus = pd.DataFrame(dataList).transpose()
        self.completeCorpus.columns = ['verse', 'tag']

        # Create shuffle splitter
        self.shuffleSplitter = StratifiedShuffleSplit(
            n_splits=shuffles, test_size=self.valratio, train_size=self.trainratio)

    # Get complete corpus
    def getCompleteCorpusDataFrame(self):
        return self.completeCorpus

    # Get complete X data of corpus
    def getCompleteX(self):
        return self.completeCorpus['verse']

    # Get complete Y data of corpus
    def getCompleteY(self):
        return self.completeCorpus['tag']

    # Get random verses from corpus
    def getRandomVerses(self, numberOfVerses, tag):
        restrictedCorpus = self.completeCorpus[self.completeCorpus['target'] == tag]
        sample = restrictedCorpus.sample(n=numberOfVerses)
        return sample

    # Get train and validation indices, i.e. run shuffle split and get all shuffle indices
    def getTrainValIndices(self):
        X = self.completeCorpus['verse']
        Y = self.completeCorpus['tag']
        return self.shuffleSplitter.split(self.completeCorpus['verse'], self.completeCorpus['tag'])


# Class representing a text vector
class TextVector:
    # Create object of TextVector
    def __init__(self, baseData, vectorType, min_df=1, ngrams=(2, 5)) -> None:
        self.baseData = baseData

        # Create vector depending on vector type
        self.vector = None
        if vectorType == VectorType.HASH:
            self.vector = HashingVectorizer(
                min_df=min_df, ngram_range=ngrams, stop_words=None)
        elif vectorType == VectorType.TFIDF:
            self.vector = TfidfVectorizer(
                min_df=min_df, ngram_range=ngrams, stop_words=None)
        else:
            self.vector = CountVectorizer(
                min_df=min_df, ngram_range=ngrams, stop_words=None)

        # Fit the vector on base data
        self.vector.fit(baseData)

    # Get transformed base data
    def getTransformedBaseData(self):
        return self.vector.transform(self.baseData)

    # Get results of transformation of new data
    def transformData(self, dataToBeTransformed):
        return self.vector.transform(dataToBeTransformed)


# Class representing a text set
class TextSet:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


# Class representing vector type as enum
class VectorType(Enum):
    COUNT = 'CountVectorizer',
    TFIDF = 'TfIdfVectorizer',
    HASH = 'HashVectorizer'
