# Main Python file for imitatio CV

from ImitatioCV import ImitatioCV, ModelType
import time

# Define algorithms to use for cross validation
analysisList = [ModelType.SVM, ModelType.DECISION_TREE, ModelType.KNN,
                ModelType.NAIVE_BAYES, ModelType.RANDOM_FOREST, ModelType.LOGISTIC_REG, ModelType.COSINE]

# Go through each algorithm
for analysis in analysisList:
    # We want to measure time performance (optional)
    time_start = time.perf_counter()

    # Define default hyperparameters
    hyperMin = 0
    hyperMax = 0
    hyperStep = 1
    ngramMin = 2
    ngramMax = 10
    folds = 20
    testSize = 0.2

    # Set hyperparameters according to concrete algorithm
    if analysis == ModelType.DECISION_TREE:
        hyperMin = 5
        hyperMax = 45
        hyperStep = 5
    elif analysis == ModelType.GRADIENT_BOOSTING:
        hyperMin = 5
        hyperMax = 45
        hyperStep = 5
    elif analysis == ModelType.KNN:
        hyperMin = 7
        hyperMax = 51
        hyperStep = 4
    elif analysis == ModelType.RANDOM_FOREST:
        hyperMin = 5  # second run 50
        hyperMax = 45  # second run 550
        hyperStep = 5  # second run 10
    elif analysis == ModelType.SVM:
        hyperMin = 1
        hyperMax = 101
        hyperStep = 10

    # Run cross validation an save results
    icv = ImitatioCV('Dante', 'Petrarca')
    icv.Run(folds, testSize, ngramMin, ngramMax,
            hyperMin, hyperMax, hyperStep, analysis)
    icv.save("CrossVal_" + analysis.name)

    # Print time consumed for current algorithm
    time_end = time.perf_counter()
    time_duration = time_end - time_start
    print("Time for " + analysis.name + ": " + f"{time_duration:.3f} seconds")
