# File with main logic of imitatio detection for one text/file
# Hidden from main application to make main application more compact

import pickle
import spacy
from spacy_syllables import SpacySyllables
from Verses import TxtText
import numpy as np


# Class representing imitatio detection
class ImitatioDetector:
    # Create object of ImitatioDetector
    def __init__(self, thresholdDante=0.8, thresholdPetrarca=0.7, linear_corr=False) -> None:
        # Load vector and model
        self.baseVocabulary = self.loadRessource(
            r"Models\CommediaCanzoniereCorpus.pkl")
        self.voter = self.loadRessource(r"Models\VoterNBLog.pkl")

        # Handle thresholds
        self.thresholdDante = thresholdDante
        self.thresholdPetrarca = thresholdPetrarca

        # Load spacy
        self.nlp = spacy.blank("it")
        self.nlp.add_pipe("syllables")

        self.linear_corr = linear_corr

    # Load ressource from pickle file
    def loadRessource(self, filepath):
        with open(filepath, 'rb') as loadfile:
            return pickle.load(loadfile)

    # Transform predict proba to labels 'Dante' or 'Petrarca'
    def transformPredictProba(self, prob1, prob2):
        if prob1 >= prob2:
            return 'Dante'
        else:
            return 'Petrarca'

    def linear_correction(self, x):
        if x < 0.5:
            return 0.8697983066164886 * x
        else:
            return 1.176990277991746 * x - 0.176990277991746

    # Analyze text, so get labels for each verse
    def analyzeText(self, filepath):
        # Get verses to be analyzed
        verses = TxtText(filepath).getSillabifiedVerses(self.nlp)
        # Get original verses (just to be able to print results in readable way)
        baseVerses = TxtText(filepath).verses

        # Reserve object for results
        result = ImitatioResult(filepath)

        # Go through verses and make prediction for each verse
        for i, line in enumerate(verses):
            # Get prediction and transorm into label
            y_pred_proba = self.voter.predict_proba(
                self.baseVocabulary.transformData([line]))
            y_pred = np.array([self.transformPredictProba(value1, value2)
                              for value1, value2 in y_pred_proba])

            # Apply linear correction if needed
            if self.linear_corr:
                y_pred_proba[0][0] = self.linear_correction(
                    y_pred_proba[0][0])
                y_pred_proba[0][1] = 1 - y_pred_proba[0][0]
                if y_pred_proba[0][0] > y_pred_proba[0][1]:
                    y_pred = "Dante"
                else:
                    y_pred = "Petrarca"

            # Handle threshold and build final vote
            if y_pred == "Dante":
                if y_pred_proba[0][0] > self.thresholdDante:
                    result.verses.append(VerseResult(
                        baseVerses[i].getOriginalVerse(), "Dante", y_pred_proba[0][0]))
                else:
                    result.verses.append(VerseResult(
                        baseVerses[i].getOriginalVerse(), "", 0))

            elif y_pred == "Petrarca":
                if y_pred_proba[0][1] > self.thresholdPetrarca:
                    result.verses.append(VerseResult(
                        baseVerses[i].getOriginalVerse(), "Petrarca", y_pred_proba[0][1]))
                else:
                    result.verses.append(VerseResult(
                        baseVerses[i].getOriginalVerse(), "", 0))

        return result


# Class representing result of complete prediction
class ImitatioResult:
    def __init__(self, text="") -> None:
        self.text = text
        self.verses = []


# Class representing result of prediction for one singfle verse
class VerseResult:
    def __init__(self, verse="", label="", probability=0) -> None:
        self.verse = verse
        self.label = label
        self.probability = probability
