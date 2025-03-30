# Draw verses to be manipulated as test verses

from Verses import Text, TxtText
import os

# Get all test verses as verse objects
commediaVerses = TxtText(r'TestVerses\CommediaImitatio.txt')
canzoniereVerses = TxtText(r'TestVerses\CanzoniereImitatio.txt')

# Split verses randomly into dev (first test run for data mismmatch) and test (second test for final testing)
randomCommediaVersesDev = commediaVerses.getRandomVerses(30)
randomCommediaVersesTest = [
    x for x in commediaVerses.verses if x not in randomCommediaVersesDev]
randomCanzoniereVersesDev = canzoniereVerses.getRandomVerses(30)
randomCanzoniereVersesTest = [
    x for x in canzoniereVerses.verses if x not in randomCanzoniereVersesDev]

# Write verses to separate files
with open(os.path.join(os.getcwd(), "TestVerses", "CommediaDevVerses" + ".txt"), "w", encoding='utf-8') as writefile:
    for verse in randomCommediaVersesDev:
        writefile.write(verse.getOriginalVerse())

with open(os.path.join(os.getcwd(), "TestVerses", "CommediaTestVerses" + ".txt"), "w", encoding='utf-8') as writefile:
    for verse in randomCommediaVersesTest:
        writefile.write(verse.getOriginalVerse())

with open(os.path.join(os.getcwd(), "TestVerses", "CanzoniereDevVerses" + ".txt"), "w", encoding='utf-8') as writefile:
    for verse in randomCanzoniereVersesDev:
        writefile.write(verse.getOriginalVerse())

with open(os.path.join(os.getcwd(), "TestVerses", "CanzoniereTestVerses" + ".txt"), "w", encoding='utf-8') as writefile:
    for verse in randomCanzoniereVersesTest:
        writefile.write(verse.getOriginalVerse())
