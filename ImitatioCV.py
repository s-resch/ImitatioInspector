# Main Python file for Imitatio CV

import PerformanceScores as ps
from Verses import Text, TextCorpus, TextVector, TextSet, VectorType, Verse
from DeltaClassifier import DeltaClassifier
from DeltaClassifier import DeltaMeasure
import os
from sklearn.metrics import classification_report
import numpy as np
import pickle
import spacy
from spacy_syllables import SpacySyllables
from itertools import combinations_with_replacement
from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# Class managing cross validation for imitatio use case
class ImitatioCV:
    # Initialize class object
    def __init__(self, targetOne=None, targetTwo=None) -> None:
        self.targetOne = targetOne
        self.targetTwo = targetTwo
        self.performManager = ps.ParamComboPerformanceManager(
            targetOne, targetTwo)

    # Run cross validation
    def Run(self, folds, testSize, minNgram, maxNgram, minHyper, maxHyper, hyperStep, method):
        # Initialize spacy
        nlp = spacy.blank("it")
        nlp.add_pipe("syllables")

        # Get/Build complete corpus
        commediaVerses = Text('Commedia.xml').getSillabifiedVerses(nlp)
        canzoniereVerses = Text('Canzoniere.xml').getSillabifiedVerses(nlp)
        canzoniereVerses = canzoniereVerses + canzoniereVerses

        commediaTags = ['Dante'] * len(commediaVerses)
        canzoniereTags = ['Petrarca'] * len(canzoniereVerses)

        corpus = TextCorpus(commediaVerses + canzoniereVerses,
                            commediaTags + canzoniereTags, 1-testSize, testSize, folds)

        repeatCounter = 0

        # Loop through folds (we used shuffleSplitter when building corpus, so we don't have a custom looping index;
        # instead we use getTrainValIndices of our corpus object)
        for i, (train_index, val_index) in enumerate(corpus.getTrainValIndices()):
            # Make concrete train and test corpora
            # Keep in mind that train_index and val_index are lists of indices
            # ShuffleSplitter, indeed, shuffles the indices in the background, so we really get random splits
            X_train = corpus.getCompleteX()[train_index]
            X_val = corpus.getCompleteX()[val_index]
            Y_train = corpus.getCompleteY()[train_index]
            Y_val = corpus.getCompleteY()[val_index]

            # Build n gram combinations
            ngramCombinations = list(
                combinations_with_replacement(range(minNgram, maxNgram + 1), 2))
            # Build hyperparameter range
            hyperVals = range(minHyper, maxHyper + 1, hyperStep)

            # Loop through ngram combinations
            for ngramCombo in ngramCombinations:
                # Build vector using current training corpus
                currentVector = TextVector(
                    X_train, VectorType.COUNT, 1, ngramCombo)

                # Loop through hyperparameters
                for hyperVal in hyperVals:

                    # Build and fit a model using current hyperparameters and current corpus vector
                    currentModel = self.getModel(method, hyperVal)
                    currentModel.fit(
                        currentVector.getTransformedBaseData(), Y_train)

                    # Predict on validation corpus (here with probabilities)
                    y_val_pred_proba = currentModel.predict_proba(
                        currentVector.transformData(X_val))

                    # Transform predicted probabilities to predicted classes
                    y_val_pred = np.array([self.transformPredictProba(
                        value1, value2) for value1, value2 in y_val_pred_proba])

                    target_names = [self.targetOne, self.targetTwo]

                    # Use classification_report to get metrics
                    classReport = classification_report(
                        Y_val, y_val_pred, target_names=target_names, output_dict=True, digits=4)

                    # Add metrics to performance manager
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.RECALL,
                                                 classReport[self.targetOne]['recall'], self.targetOne)
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.RECALL,
                                                 classReport[self.targetTwo]['recall'], self.targetTwo)
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.PRECISION,
                                                 classReport[self.targetOne]['precision'], self.targetOne)
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.PRECISION,
                                                 classReport[self.targetTwo]['precision'], self.targetTwo)
                    self.performManager.addValue(
                        ngramCombo, hyperVal, ps.PerfScoreTitle.F1, classReport[self.targetOne]['f1-score'], self.targetOne)
                    self.performManager.addValue(
                        ngramCombo, hyperVal, ps.PerfScoreTitle.F1, classReport[self.targetTwo]['f1-score'], self.targetTwo)
                    self.performManager.addValue(
                        ngramCombo, hyperVal, ps.PerfScoreTitle.ACCURACY, classReport['accuracy'])
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.AVG_PRECISION,
                                                 classReport['macro avg']['precision'])
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.AVG_RECALL,
                                                 classReport['macro avg']['recall'])
                    self.performManager.addValue(
                        ngramCombo, hyperVal, ps.PerfScoreTitle.AVG_F1, classReport['macro avg']['f1-score'])
                    self.performManager.addValue(
                        ngramCombo, hyperVal, ps.PerfScoreTitle.WEIGHT_RECALL, classReport['weighted avg']['recall'])
                    self.performManager.addValue(
                        ngramCombo, hyperVal, ps.PerfScoreTitle.WEIGHT_PRECISION, classReport['weighted avg']['precision'])
                    self.performManager.addValue(ngramCombo, hyperVal, ps.PerfScoreTitle.WEIGHT_F1,
                                                 classReport['weighted avg']['f1-score'])

                    # Increment repeat counter
                    repeatCounter = repeatCounter + 1

                    # if (npExpected[0] != npPredicted[0]):
                    #     self.performManager.addFailureRun(mfw, cull, testFile)

        # Documentation: Print number of repetitions
        print("Repetions for " + method.name + ": " + str(repeatCounter))

    # Transform predicted probabilities to predicted classes
    def transformPredictProba(self, prob1, prob2):
        if prob1 >= prob2:
            return self.targetOne
        else:
            return self.targetTwo

    # Get maximum scores from performance manager (so we get the max settings for each score)
    def getMaxScores(self):
        returnScores = [
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.PRECISION, self.targetOne),
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.RECALL, self.targetOne),
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.F1, self.targetOne),
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.PRECISION, self.targetTwo),
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.RECALL, self.targetTwo),
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.F1, self.targetTwo),
            self.performManager.getMaxScore(ps.PerfScoreTitle.ACCURACY),
            self.performManager.getMaxScore(ps.PerfScoreTitle.AVG_PRECISION),
            self.performManager.getMaxScore(ps.PerfScoreTitle.AVG_RECALL),
            self.performManager.getMaxScore(ps.PerfScoreTitle.AVG_F1),
            self.performManager.getMaxScore(ps.PerfScoreTitle.WEIGHT_RECALL),
            self.performManager.getMaxScore(
                ps.PerfScoreTitle.WEIGHT_PRECISION),
            self.performManager.getMaxScore(ps.PerfScoreTitle.WEIGHT_F1)
        ]

        return returnScores

    # Get best score from performance manager (best overall score)
    def getBestScore(self):
        return self.performManager.getBestScore()

    # Get macro average precision for all scores
    def getMacroAveragePrecisionAllScores(self):
        return self.performManager.getMacroAveragePrecisionAllScores()

    # Get macro average recall for all scores
    def getMacroAverageRecallAllScores(self):
        return self.performManager.getMacroAverageRecallAllScores()

    # Get macro average F1 score for all scores
    def getMacroAverageF1AllScores(self):
        return self.performManager.getMacroAverageF1AllScores()

    # Get macro average accuracy for all scores
    def getMacroAverageAccuracyAllScores(self):
        return self.performManager.getMacroAverageAccuracyAllScores()

    # Get model object based on model type
    def getModel(self, modelType, hyper):
        model = None
        if modelType == ModelType.KNN:
            model = KNeighborsClassifier(n_neighbors=hyper)
        elif modelType == ModelType.DECISION_TREE:
            model = DecisionTreeClassifier(max_depth=hyper, random_state=11)
        elif modelType == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=hyper, random_state=11)
        elif modelType == ModelType.NAIVE_BAYES:
            model = MultinomialNB()
        elif modelType == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=hyper, random_state=11)
        elif model == ModelType.SVM:
            model = SVC(probability=True, C=hyper, random_state=11)
        elif modelType == ModelType.BURROW:
            model = DeltaClassifier(delta=DeltaMeasure.BURROW)
        elif modelType == ModelType.COSINE:
            model = DeltaClassifier(delta=DeltaMeasure.COSINE)
        else:
            model = LogisticRegression(
                multi_class='multinomial', max_iter=10000, random_state=11)

        return model

    # Save results into a pickle
    def save(self, filename):
        filepath = os.path.join(os.getcwd(), filename + ".pkl")
        with open(filepath, 'wb') as savefile:
            pickle.dump(self.performManager, savefile)

    # Load results from a pickle
    def load(self, filepath):
        with open(filepath, 'rb') as loadfile:
            self.performManager = pickle.load(loadfile)


# Model Types
class ModelType(Enum):
    LOGISTIC_REG = 'Logistic Regression',
    KNN = 'knn'
    NAIVE_BAYES = 'Naive Bayes',
    DECISION_TREE = 'Decision Tree',
    RANDOM_FOREST = 'Random Forest',
    GRADIENT_BOOSTING = 'Gradient Boosting',
    SVM = 'SVM',
    BURROW = 'Burrow',
    COSINE = 'Cosine'
