# File with real-world application to detect imitatio for several texts/file

from ImitatioClassifier import ImitatioDetector, ImitatioResult
import argparse
import warnings
from PdfExport import PdfExport, ResultSummary


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Imitatio Inspector")

    # Add arguments
    # Add an argument to accept multiple files
    parser.add_argument(
        "files",
        nargs="+",
        help="List of files to be analyzed"
    )
    parser.add_argument(
        "--dante",
        type=float,
        default=0.8,
        help="Threshold for Dante label, e.g. 0.8"
    )
    parser.add_argument(
        "--petrarca",
        type=float,
        default=0.8,
        help="Threshold for Petrarca label, e.g. 0.8"
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        default=False,
        help="Show details in console"
    )
    parser.add_argument(
        "--show_warnings",
        action="store_true",
        default=False,
        help="Show warnings"
    )
    parser.add_argument(
        "--linear_corr",
        action="store_true",
        default=False,
        help="Perform linear correction of internal bias"
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        default=False,
        help="Create a PDF report"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    if not args.show_warnings:
        warnings.filterwarnings("ignore")

    # List of files to analyze
    filesToAnalyze = args.files

    # Create object of ImitatioDetector
    detector = ImitatioDetector(
        thresholdDante=args.dante, thresholdPetrarca=args.petrarca, linear_corr=args.linear_corr)

    # Analyze texts and collect results for all texts
    results = []
    for file in filesToAnalyze:
        results.append(detector.analyzeText(file))

    # Print results (for each file/text)
    for result in results:
        numberDante = len([x for x in result.verses if x.label == "Dante"])
        numberPetrarca = len(
            [x for x in result.verses if x.label == "Petrarca"])
        numberBase = len(result.verses)

        # Print text title
        print(result.text)

        # Print summary
        print("")
        print("Dante: " + str(numberDante))
        print("Petrarca: " + str(numberPetrarca))
        print("Dante relative: " + str(numberDante/numberBase))
        print("Petrarca relative: " + str(numberPetrarca/numberBase))

        # Print details (prediction for each single verse of text)
        if (args.show_details):
            print("")
            print("")
            print("Details: ")
        if args.linear_corr:
            filename = "Results_" + result.text + "_lc_" + \
                str(args.dante).replace('.', '') + "_" + \
                str(args.petrarca).replace('.', '') + ".txt"
        else:
            filename = "Results_" + result.text + \
                str(args.dante).replace('.', '') + "_" + \
                str(args.petrarca).replace('.', '') + ".txt"
        with open(filename, "w", encoding='utf-8') as writefile:
            for verse in result.verses:
                if (args.show_details):
                    print(verse.verse.rstrip() + " - " + verse.label +
                          " - " + str(verse.probability))
                writefile.write(verse.verse.rstrip() + " - " + verse.label +
                                " - " + str(verse.probability) + "\n")

        # Print summary again if we showed details
        if (args.show_details):
            print("")
            print("")
            print("Dante: " + str(numberDante))
            print("Petrarca: " + str(numberPetrarca))
            print("Dante relative: " + str(numberDante/numberBase))
            print("Petrarca relative: " + str(numberPetrarca/numberBase))
        print("")
        print("____________________________________________________________________")

        if args.pdf:
            # Create a PDF report
            pdfExport = PdfExport()
            summary = ResultSummary(numberDante, numberPetrarca,
                                    numberDante/numberBase, numberPetrarca/numberBase)
            pdfExport.createResultPdf(
                result.text, result.verses, "not indicated",
                summary, args.dante, args.petrarca, args.linear_corr)


if __name__ == "__main__":
    main()
