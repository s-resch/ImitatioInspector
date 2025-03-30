# File to frun tests (dev and (final) test stage) on selected (pre-trained via ModelTrain.py) models

import spacy
from spacy_syllables import SpacySyllables
from Verses import Text, TxtText, TextCorpus, TextVector, TextSet, VectorType, Verse
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
import os
import pickle
from DeltaClassifier import DeltaClassifier, DeltaMeasure
from statistics import harmonic_mean


# Transform predict proba to labels 'Dante' or 'Petrarca'
def transformPredictProba(prob1, prob2):
    # We need a forced decision because otherwise classification_report() would throw an error
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

# Load verses to use for tests from files
# "TestVerses\CommediaTestVerses.txt"
commediaVerseSource = r"TestVerses\CommediaTestVerses.txt"
# "TestVerses\CanzoniereTestVerses.txt"
canzoniereVerseSource = r"TestVerses\CanzoniereTestVerses.txt"

# Transform test verses into verse objects
commediaDevVerses = TxtText(
    commediaVerseSource).getSillabifiedVerses(nlp)
canzoniereDevVerses = TxtText(
    canzoniereVerseSource).getSillabifiedVerses(nlp)

# Set labels
commediaDevTags = ['Dante'] * len(commediaDevVerses)
canzoniereDevTags = ['Petrarca'] * len(canzoniereDevVerses)

# Create corpus (pandas)
devCorpus = [commediaDevVerses + canzoniereDevVerses,
             commediaDevTags + canzoniereDevTags]

dfDev = pd.DataFrame(devCorpus).transpose()
dfDev.columns = ['verse', 'tag']

# Split into X and Y
X_dev = dfDev['verse']
Y_dev = dfDev['tag']

# Load vector and models from file
textVector = load(r"Models\CommediaCanzoniereCorpus.pkl")
textVectorCosine = load(r"Models\CommediaCanzoniereCorpus_Cosine.pkl")

NaiveBayesModel = load(r"Models\NB_BaseModel.pkl")
SvmModel = load(r"Models\SVM_BaseModel.pkl")
LogRegModel = load(r"Models\LogReg_BaseModel.pkl")

voter = load(r"Models\VoterNBLog.pkl")
voterSVM = load(r"Models\VoterNBLogSVM.pkl")

cosine = load(r"Models\CosineDelta_BaseModel.pkl")

# Define target names
target_names = ['Dante', 'Petrarca']


# Make predictions for single models
# Naive Bayes
y_val_pred_proba_naive = NaiveBayesModel.predict_proba(
    textVector.transformData(X_dev))

y_val_pred_naive = np.array([transformPredictProba(
    value1, value2) for value1, value2 in y_val_pred_proba_naive])

# SVM
y_val_pred_proba_svm = SvmModel.predict_proba(
    textVector.transformData(X_dev))

y_val_pred_svm = np.array([transformPredictProba(
    value1, value2) for value1, value2 in y_val_pred_proba_svm])

# Logistic Regression
y_val_pred_proba_log = LogRegModel.predict_proba(
    textVector.transformData(X_dev))

y_val_pred_log = np.array([transformPredictProba(
    value1, value2) for value1, value2 in y_val_pred_proba_log])


# Get metrics for each model using classification_report
classReportNaive = classification_report(
    Y_dev, y_val_pred_naive, target_names=target_names, output_dict=True, digits=4)

classReportSVM = classification_report(
    Y_dev, y_val_pred_svm, target_names=target_names, output_dict=True, digits=4)

classReportLog = classification_report(
    Y_dev, y_val_pred_log, target_names=target_names, output_dict=True, digits=4)


# Make predictions for voting models and get metrics
# Voter using Naive Bayes and Logistic Regression
y_val_pred_proba_voter = voter.predict_proba(
    textVector.transformData(X_dev))

y_val_pred_voter = np.array([transformPredictProba(
    value1, value2) for value1, value2 in y_val_pred_proba_voter])

classReportVoter = classification_report(
    Y_dev, y_val_pred_voter, target_names=target_names, output_dict=True, digits=4)

# Voter using Naive Bayes, Logistic Regression and SVM
y_val_pred_proba_voter_svm = voterSVM.predict_proba(
    textVector.transformData(X_dev))

y_val_pred_voter_svm = np.array([transformPredictProba(
    value1, value2) for value1, value2 in y_val_pred_proba_voter_svm])

classReportVoterSVM = classification_report(
    Y_dev, y_val_pred_voter_svm, target_names=target_names, output_dict=True, digits=4)


# Make predictions for delta
# Cosine delta
y_val_pred_proba_cosine = cosine.predict_proba(
    textVectorCosine.transformData(X_dev))

y_val_pred_cosine = np.array([transformPredictProba(
    value1, value2) for value1, value2 in y_val_pred_proba_cosine])

classReportVoterCosine = classification_report(
    Y_dev, y_val_pred_cosine, target_names=target_names, output_dict=True, digits=4)


# Print metrics
print("Voter F1: " + str(classReportVoter['macro avg']['f1-score']))
print("Voter Accuracy: " + str(classReportVoter['accuracy']))
print("Voter F1_Acc_Harm: " +
      str(harmonic_mean([classReportVoter['macro avg']['f1-score'], classReportVoter['accuracy']])))
print("")
print("Voter SVM F1: " + str(classReportVoterSVM['macro avg']['f1-score']))
print("Voter SVM Accuracy: " + str(classReportVoterSVM['accuracy']))
print("Voter SVM F1_Acc_Harm: " + str(harmonic_mean(
    [classReportVoterSVM['macro avg']['f1-score'], classReportVoterSVM['accuracy']])))
print("")
print("Naive Bayes F1: " + str(classReportNaive['macro avg']['f1-score']))
print("Naive Bayes Accuracy: " + str(classReportNaive['accuracy']))
print("Naive Bayes F1_Acc_Harm: " +
      str(harmonic_mean([classReportNaive['macro avg']['f1-score'], classReportNaive['accuracy']])))
print("")
print("SVM F1: " + str(classReportSVM['macro avg']['f1-score']))
print("SVM Accuracy: " + str(classReportSVM['accuracy']))
print("SVM F1_Acc_Harm: " +
      str(harmonic_mean([classReportSVM['macro avg']['f1-score'], classReportSVM['accuracy']])))
print("")
print("Log Reg F1: " + str(classReportLog['macro avg']['f1-score']))
print("Log Reg Accuracy: " + str(classReportLog['accuracy']))
print("Log Reg F1_Acc_Harm: " +
      str(harmonic_mean([classReportLog['macro avg']['f1-score'], classReportLog['accuracy']])))
print("")
print("Cosine F1: " + str(classReportVoterCosine['macro avg']['f1-score']))
print("Cosine Accuracy: " + str(classReportVoterCosine['accuracy']))
print("Cosine F1_Acc_Harm: " + str(harmonic_mean(
    [classReportVoterCosine['macro avg']['f1-score'], classReportVoterCosine['accuracy']])))
print("")
