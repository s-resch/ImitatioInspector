# _ImitatioInspector_: A Prototype Machine Learning Tool Regarding _Imitatio Dantis_ and _Imitatio Petrarcae_

This repository contains code and material for an application for detecting _imitatio Dantis_ and _imitatio Petrarcae_ using machine learning algorithms as described in Sascha Resch's essay "_ImitatioInspector_: A Prototype Machine Learning Tool Regarding _Imitatio Dantis_ and _Imitatio Petrarcae_", published in _xxx_ xx (202x). See there for a detailed description of textual corpus, test results and methodic considerations.

# Usage

Make sure that you have already installed all packages needed. You can find them in `requirements.txt`(this file was generated using `pipreqs`using `pip install pipreqs`). As soon as you have installed all packages, you can call the app using command line/terminal:

```bash
python ImitatioInspectorApp.py [files] [options]
```

## Arguments

- **files**: (Required) A list of files to be analyzed. You can specify multiple files by separating them with spaces.

## Optional Arguments

- `--dante`:

  - **Type**: Float
  - **Default**: 0.8
  - **Description**: Sets the threshold for the Dante label. For example, you can specify `--dante=0.8` to set the threshold to 0.8.

- `--petrarca`:

  - **Type**: Float
  - **Default**: 0.8
  - **Description**: Sets the threshold for the Petrarca label. For example, you can specify `--petrarca=0.8` to set the threshold to 0.8.

- `--show_details`:

  - **Type**: Boolean (Flag)
  - **Default**: False
  - **Description**: If set, the application will display detailed results, i.e. results for all verses, in the console.

- `--show_warnings`:

  - **Type**: Boolean (Flag)
  - **Default**: False
  - **Description**: If set, the application will display warnings in the console.

- `--linear_corr`:

  - **Type**: Boolean (Flag)
  - **Default**: False
  - **Description**: If set, the application will perform a linear correction of internal bias.

- `--pdf`:
  - **Type**: Boolean (Flag)
  - **Default**: False
  - **Description**: If set, the application will produce a pdf containing the results.

## Example

```bash
python ImitatioInspectorApp.py "file1.txt" "file2.txt" --dante=0.85 --petrarca=0.85 --show_details --show_warnings --linear_corr --pdf
```

This command will analyze `file1.txt` and `file2.txt` with custom thresholds for Dante and Petrarca labels, display detailed information and warnings, and perform a linear correction of internal bias. The results will be saved using PDF format.

# Roadmap

(Minor) Future improvements include:

- Refactoring of code (removing unused imports)
- Refactoring of code (separate Python scripts into folders and realize relative import)
- Writing of further comments

For major improvements and general future developments see the essay "_ImitatioInspector_: A Prototype Machine Learning Tool Regarding _Imitatio Dantis_ and _Imitatio Petrarcae_"

# Copyright

To run the analysis the following software and resources were used (as indicated in the essay "_ImitatioInspector_: A Prototype Machine Learning Tool Regarding _Imitatio Dantis_ and _Imitatio Petrarcae_"):

- Explosion AI GmbH 2016: spaCy, https://spacy.io/usage/facts-figures. [last access: 30.03.25].
- Hunter, J. D. 2007: „Matplotlib: A 2D graphics environment“, in: Computing in Science & Engineering, vol. 9, 3, p. 90–95.
- La Sapienza Università di Roma (Ed.) 2003: Biblioteca Italiana, Rome, http://www.bibliotecaitaliana.it/. [last access: 30.03.25].
- NumFocus 2007–: scikit-learn, https://scikit-learn.org/stable/user_guide.html. [last access: 30.03.25].
- NumFocus 2008–: Pandas, https://pandas.pydata.org/docs/user_guide/index.html#user-guide. [last access: 30.03.25].
- Python Software Foundation (2001): Python Language Reference, version 3.9.7. https://docs.python.org/3.9/. [last access: 30.03.25].

The fonts included to generate PDF files with Unicode characters are taken from PyPDF repository: https://github.com/reingart/pyfpdf/releases. See also here for the general license of DejaVu fonts: https://dejavu-fonts.github.io/License.html

Due to copyright concerns, we don't publish the TEI XML-files of Dante's _Comedy_ and Petrarch's _Canzoniere_. Please refer to _Biblioteca Italiana_ for these files.
