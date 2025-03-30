# Performance classes
from enum import Enum
import statistics


# Class managing parameter combination performance
class ParamComboPerformanceManager:
    # Initialize ParamComboPerformanceManager class object
    def __init__(self, targetOne=None, targetTwo=None) -> None:
        self.__paramterComboPerformances = []
        self.__failureRuns = []
        self.__targetOne = targetOne
        self.__targetTwo = targetTwo

    # Add value to manager
    def addValue(self, ngram, hyper, valueTitle, value, target=None):
        # Get existing or create new ParamComboPerformance
        paramCombo = next(
            (x for x in self.__paramterComboPerformances if x.ngram == ngram and x.hyper == hyper), None)
        if paramCombo == None:
            paramCombo = ParameterComboPerformance(
                ngram, hyper, self.__targetOne, self.__targetTwo)
            self.__paramterComboPerformances.append(paramCombo)
        # Add value to paramCombo
        paramCombo.addValue(valueTitle, value, target)

    # Add failure run
    def addFailureRun(self, ngram, hyper, textTitle):
        self.__failureRuns.append(FailureRun(textTitle, ngram, hyper))

    # Get max score for concrete metric (valueTitle)
    def getMaxScore(self, valueTitle, target=None):
        maxValue = 0
        maxCombo = None
        for x in self.__paramterComboPerformances:
            mean = x.getArithmeticMean(valueTitle, target)
            if mean >= maxValue:
                maxValue = mean
                maxCombo = x

        return [valueTitle, target, maxCombo]

    # Get best overall score
    def getBestScore(self):
        maxScore = 0
        maxCombo = None
        for x in self.__paramterComboPerformances:
            # sum = x.getMeanScoreSum()
            sum = statistics.harmonic_mean([x.getArithmeticMean(
                PerfScoreTitle.ACCURACY), x.getArithmeticMean(PerfScoreTitle.AVG_F1)])
            if sum >= maxScore:
                maxScore = sum
                maxCombo = x

        return maxCombo

    # Get harmonic mean of accuracy and F1 for one metric
    def getSingleScoreAccF1HarmonicMean(self, ngram, hyper):
        harmonicMean = 0
        for x in self.__paramterComboPerformances:
            if x.ngram == ngram and x.hyper == hyper:
                accuracy = x.getSingleScore(
                    PerfScoreTitle.ACCURACY).getArithmeticMean()
                f1 = x.getSingleScore(
                    PerfScoreTitle.AVG_F1).getArithmeticMean()
                harmonicMean = statistics.harmonic_mean([accuracy, f1])

        return harmonicMean

    # Get all scores
    def getAllScores(self):
        return self.__paramterComboPerformances

    # Get failure runs
    def getFailureRuns(self):
        return self.__failureRuns

    # Get scores for one metric (valueTitle)
    def getScores(self, valueTitle, target=None):
        returnScores = []
        for x in self.__paramterComboPerformances:
            returnScores.append(x.getScore(valueTitle, target))

        return returnScores

    # Get macro average precision for all scores
    def getMacroAveragePrecisionAllScores(self):
        precisionList = [x.getSingleScore(PerfScoreTitle.AVG_PRECISION).getArithmeticMean()
                         for x in self.__paramterComboPerformances]
        return statistics.mean(precisionList)

    # Get macro average weighted precision for all scores
    def getMacroAveragePrecisionWeightedAllScores(self):
        precisionList = [2 ** (x.getSingleScore(PerfScoreTitle.AVG_PRECISION).getArithmeticMean())
                         for x in self.__paramterComboPerformances]
        return statistics.mean(precisionList)

    # Get macro average recall for all scores
    def getMacroAverageRecallAllScores(self):
        recallList = [x.getSingleScore(PerfScoreTitle.AVG_RECALL).getArithmeticMean()
                      for x in self.__paramterComboPerformances]
        return statistics.mean(recallList)

    # Get macro average weighted recall for all scores
    def getMacroAverageRecallWeightedAllScores(self):
        recallList = [2 ** (x.getSingleScore(PerfScoreTitle.AVG_RECALL).getArithmeticMean())
                      for x in self.__paramterComboPerformances]
        return statistics.mean(recallList)

    # Get macro average F1 score for all scores
    def getMacroAverageF1AllScores(self):
        precisionAvg = self.getMacroAveragePrecisionAllScores()
        recallAvg = self.getMacroAverageRecallAllScores()
        return statistics.harmonic_mean([precisionAvg, recallAvg])

    # Get macro average weighted F1 score for all scores
    def getMacroAverageF1WeightedAllScores(self):
        precisionAvg = self.getMacroAveragePrecisionWeightedAllScores()
        recallAvg = self.getMacroAverageRecallWeightedAllScores()
        return statistics.harmonic_mean([precisionAvg, recallAvg])

    # Get macro average accuracy for all scores
    def getMacroAverageAccuracyAllScores(self):
        accuracyList = [x.getSingleScore(PerfScoreTitle.ACCURACY).getArithmeticMean()
                        for x in self.__paramterComboPerformances]
        return statistics.mean(accuracyList)

    # Get macro average weighted accuracy for all scores
    def getMacroAverageAccuracyWeightedAllScores(self):
        accuracyList = [2 ** (x.getSingleScore(PerfScoreTitle.ACCURACY).getArithmeticMean())
                        for x in self.__paramterComboPerformances]
        return statistics.mean(accuracyList)

    # Get accuracy for one combo
    def getSingleComboAccuracy(self, ngram, hyper):
        accuracy = 0
        for x in self.__paramterComboPerformances:
            if x.ngram == ngram and x.hyper == hyper:
                accuracy = x.getSingleScore(
                    PerfScoreTitle.ACCURACY).getArithmeticMean()

        return accuracy


# Class encapsulating a single parameter combination and its scores
class ParameterComboPerformance:
    # Create object of ParameterComboPerformance
    def __init__(self, ngram, hyper, targetOne=None, targetTwo=None) -> None:
        self.ngram = ngram
        self.hyper = hyper
        self.performanceScores = [
            PerformanceScore(PerfScoreTitle.RECALL, targetOne),
            PerformanceScore(PerfScoreTitle.PRECISION, targetOne),
            PerformanceScore(PerfScoreTitle.F1, targetOne),
            PerformanceScore(PerfScoreTitle.RECALL, targetTwo),
            PerformanceScore(PerfScoreTitle.PRECISION, targetTwo),
            PerformanceScore(PerfScoreTitle.F1, targetTwo),
            PerformanceScore(PerfScoreTitle.ACCURACY),
            PerformanceScore(PerfScoreTitle.AVG_RECALL),
            PerformanceScore(PerfScoreTitle.AVG_PRECISION),
            PerformanceScore(PerfScoreTitle.AVG_F1),
            PerformanceScore(PerfScoreTitle.WEIGHT_RECALL),
            PerformanceScore(PerfScoreTitle.WEIGHT_PRECISION),
            PerformanceScore(PerfScoreTitle.WEIGHT_F1)
        ]

    # Add new value
    def addValue(self, valueTitle, value, target=None):
        # Get fitting score and add value if it exists
        score = next((x for x in self.performanceScores if x.title ==
                     valueTitle and x.target == target), None)
        if score != None:
            score.addValue(value)

    # Get arithmetic mean for one metric (valueTitle)
    def getArithmeticMean(self, valueTitle, target=None):
        score = next((x for x in self.performanceScores if x.title ==
                     valueTitle and x.target == target), None)
        if score != None:
            return score.getArithmeticMean()
        else:
            return None

    # Get arithmetic mean of all metrics as sum
    def getMeanScoreSum(self):
        sumValue = sum([x.getArithmeticMean() for x in self.performanceScores])
        return sumValue

    # Get harmonic mean of accuracy and F1
    def getHarmonicMeanAccuracyF1(self):
        relevantValues = [x.getArithmeticMean() for x in self.performanceScores if x.title ==
                          PerfScoreTitle.ACCURACY or x.title == PerfScoreTitle.AVG_F1]
        return statistics.harmonic_mean(relevantValues)

    # Get all metrics
    def getAllScores(self):
        return self.performanceScores

    # Get single metric (title)
    def getSingleScore(self, title, target=None):
        score = next(x for x in self.performanceScores if x.title ==
                     title and x.target == target)
        return score


# Class representing a single performance score (metric)
class PerformanceScore:
    # Create object of PerformanceScore
    def __init__(self, title, target=None) -> None:
        self.title = title
        self.target = target
        self.values = []

    # Add new value (of one CV run)
    def addValue(self, value):
        self.values.append(value)

    # Get arithmetic mean (of all CV runs)
    def getArithmeticMean(self):
        return sum(self.values)/len(self.values)

    # Get raw values
    def getRawValues(self):
        return self.values


# Class representing a single failure run
class FailureRun:
    # Create object of FailureRun
    def __init__(self, title, ngram, hyper) -> None:
        self.title = title
        self.ngram = ngram
        self.hyper = hyper


# Enum for performance score (metric) titles
class PerfScoreTitle(Enum):
    RECALL = 'recall',
    PRECISION = 'precision',
    F1 = 'f1',
    AVG_RECALL = 'avgRecall',
    AVG_PRECISION = 'avgPrecision',
    AVG_F1 = 'avgF1',
    WEIGHT_RECALL = 'weightAvgRecall',
    WEIGHT_PRECISION = 'weightAvgPrecision',
    WEIGHT_F1 = 'weightAvgF1',
    ACCURACY = 'accuracy'
