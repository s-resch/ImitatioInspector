# File for final model training

import spacy
from spacy_syllables import SpacySyllables
from Verses import Text, TxtText, TextCorpus, TextVector, TextSet, VectorType, Verse
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from DeltaClassifier import DeltaClassifier, DeltaMeasure
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import os
import pickle


# Transform predict proba to labels 'Dante' or 'Petrarca'
def transformPredictProba(prob1, prob2):
    if prob1 >= prob2:
        return 'Dante'
    else:
        return 'Petrarca'


# Save model/vector into pickle file
def save(model, filename):
    filepath = os.path.join(os.getcwd(), filename + ".pkl")
    with open(filepath, 'wb') as savefile:
        pickle.dump(model, savefile)


# Load model/vector from pickle file
def load(filepath):
    with open(filepath, 'rb') as loadfile:
        return pickle.load(loadfile)


# Load spacy
nlp = spacy.blank("it")
nlp.add_pipe("syllables")

# Get verses from Commedia and Canzoniere
commediaVerses = Text('Commedia.xml').getSillabifiedVerses(nlp)
canzoniereVerses = Text('Canzoniere.xml').getSillabifiedVerses(nlp)
canzoniereVerses = canzoniereVerses + canzoniereVerses

# Set labels
commediaTags = ['Dante'] * len(commediaVerses)
canzoniereTags = ['Petrarca'] * len(canzoniereVerses)

# Build corpus (pandas)
corpus = [commediaVerses + canzoniereVerses, commediaTags + canzoniereTags]
df = pd.DataFrame(corpus).transpose()
df.columns = ['verse', 'tag']

# Split data and labels
X = df['verse']
Y = df['tag']

# Build and save vector
textVector = TextVector(X, VectorType.COUNT, 1, (2, 5))
save(textVector, r"Models\CommediaCanzoniereCorpus")

textVectorCosine = TextVector(X, VectorType.COUNT, 1, (5, 10))
save(textVectorCosine, r"Models\CommediaCanzoniereCorpus_Cosine")

# Set up single models
NaiveBayesModel = MultinomialNB()
SvmModel = SVC(probability=True, random_state=11)
LogRegModel = LogisticRegression(
    multi_class='multinomial', max_iter=10000, random_state=11)

CosineModel = DeltaClassifier(delta=DeltaMeasure.COSINE)

# Set up models for voting
NaiveBayesModelVoter = MultinomialNB()
LogRegModelVoter = LogisticRegression(
    multi_class='multinomial', max_iter=10000, random_state=11)

NaiveBayesModelVoterSVM = MultinomialNB()
SvmModelVoterSVM = SVC(probability=True, random_state=11)
LogRegModelVoterSVM = LogisticRegression(
    multi_class='multinomial', max_iter=10000, random_state=11)

# Set up voting models
voterNBLog = VotingClassifier(
    estimators=[('lr', LogRegModelVoter), ('nb', NaiveBayesModelVoter)], voting='soft')

voterNBLogSVM = VotingClassifier(
    estimators=[('lr', LogRegModelVoterSVM), ('nb', NaiveBayesModelVoterSVM), ('svm', SvmModelVoterSVM)], voting='soft')

# Train single models and save them into pickle file
NaiveBayesModel.fit(textVector.getTransformedBaseData(), Y)
save(NaiveBayesModel, r"Models\NB_BaseModel")
SvmModel.fit(textVector.getTransformedBaseData(), Y)
save(SvmModel, r"Models\SVM_BaseModel")
LogRegModel.fit(textVector.getTransformedBaseData(), Y)
save(LogRegModel, r"Models\LogReg_BaseModel")

CosineModel.fit(textVectorCosine.getTransformedBaseData(), Y)
save(CosineModel, r"Models\CosineDelta_BaseModel")


# Train voting models and save them into pickle file
voterNBLogSVM.fit(textVector.getTransformedBaseData(), Y)
save(voterNBLogSVM, r"Models\VoterNBLogSVM")
voterNBLog.fit(textVector.getTransformedBaseData(), Y)
save(voterNBLog, r"Models\VoterNBLog")
