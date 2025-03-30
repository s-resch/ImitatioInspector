# Python file for analysis of CV data (in pickle files)

from itertools import combinations_with_replacement
from ImitatioCV import ImitatioCV
import os
import matplotlib.pyplot as pl

# Define path of result pickle files and get list of files
filepath = r"CV_Results"
listOfObjects = os.listdir(filepath)
listOfFiles = [x for x in listOfObjects if os.path.isfile(
    os.path.join(filepath, x))]


# Evaluate each pickle file
for file in listOfFiles:
    # Only load pickle files
    if "pkl" in file:
        # Load pickle file
        icv = ImitatioCV('Dante', 'Petrarca')
        icv.load(os.path.join(filepath, file))

        # Evaluate: Get max scores and best overall score
        maxScore = icv.getMaxScores()
        bestScore = icv.getBestScore()

        # Reserve file name for evaluation results
        writeFileName = file.replace(".pkl", "_evaluate.txt")

        twoTwoValues = []
        xLaberls = []

        # Write results into file
        with open(os.path.join(filepath, writeFileName), "w") as writefile:
            for x in maxScore:
                writefile.write("Measure: " + str(x[0]) + " - Target: " + str(x[1]) + " - ParamterCombo: " +
                                str(x[2].ngram) + " Ngrams @ " + str(x[2].hyper) + " Hyper --> " + str(x[2].getSingleScore(x[0], x[1]).getArithmeticMean()) + "\n\n")

            writefile.write("Best score: " + str(bestScore.ngram) +
                            " Ngrams @ " + str(bestScore.hyper) + " Hyper --> " + str(bestScore.getHarmonicMeanAccuracyF1()) + "\n\n")

            writefile.write("Overall Precision: " +
                            str(icv.getMacroAveragePrecisionAllScores()) + "\n")
            writefile.write("Overall Recall: " +
                            str(icv.getMacroAverageRecallAllScores()) + "\n")
            writefile.write("Overall F1: " +
                            str(icv.getMacroAverageF1AllScores()) + "\n")
            writefile.write("Overall Accuracy: " +
                            str(icv.getMacroAverageAccuracyAllScores()) + "\n")

            
            # Detailed analysis START
            #
            # if "COSINE" in writeFileName:
            #   writefile.write(
            #       "AccF1 Mean @ 2,5: " + str(icv.performManager.getSingleScoreAccF1HarmonicMean((2, 5), 0)) + "\n")
            #
            # elif "SVM" in writeFileName:
            #     for i in range(1, 101, 10):
            #         writefile.write(
            #             "Hyper" + str(i) + "Accuracy: " + str(icv.performManager.getSingleComboAccuracy((2, 5), i)) + "\n")
            #
            # elif "KNN" in writeFileName:
            #     for i in range(7, 51, 4):
            #
            #         writefile.write(
            #             "Hyper" + str(i) + "Accuracy: " + str(icv.performManager.getSingleComboAccuracy((2, 2), i)) + "\n")
            #
            # elif "RANDOM" in writeFileName or "DECISION" in writeFileName:
            #     for i in range(5, 61, 5):
            #
            #         twoTwoValues.append(
            #             icv.performManager.getSingleScoreAccF1HarmonicMean((2, 2), i))
            #         xLabels.append(i)
            #         writefile.write(
            #             "Hyper" + str(i) + "Accuracy: " + str(icv.performManager.getSingleScoreAccF1HarmonicMean((2, 2), i)) + "\n")
            #     xi = list(range(len(xLabels)))
            #     pl.plot(xi, twoTwoValues, marker='o', color='b', label='Square')
            #     pl.xlabel('Number of estimators')
            #     pl.ylabel('Score (F1-accuracy combined)')
            #     pl.ylim(0.5, 1)
            #     pl.xticks(xi, xLabels, rotation=45)
            #     pl.plot(twoTwoValues)
            #     pl.show()
            #
            # elif "NAIVE" in writeFileName or "LOGISTIC" in writeFileName:
            #     writefile.write(
            #         "AccF1 Mean @ 2,5: " + str(icv.performManager.getSingleScoreAccF1HarmonicMean((2, 5), 0)) + "\n")
            #
            # Detailed analysis END
