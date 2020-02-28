import spacy
import logging
import texttable
from pathlib import Path
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_md") 
#nlp = spacy.load("en_trf_bertbaseuncased_lg")

logging.basicConfig(filename='logs.log',level=logging.INFO)

def getSynset(word):
    return wn.synsets(word.rstrip())[0]
    
def calcSpacySimilarity(w1,w2):
    token1 = nlp(w1)
    token2 = nlp(w2)
    return token1.similarity(token2)

if __name__ == "__main__":

    with open(Path("testfile.txt"), 'r') as f:
        data = f.readlines()

    mySentences = [i.rstrip('\n') for i in data]

    rows = [['Words','WN Sim','Spacy Sim']]
    
    for i in mySentences:
        syn1 = getSynset(i)
        for j in mySentences:
            syn2 = getSynset(j)
            wnSim = syn1.path_similarity(syn2)
            spacySim = calcSpacySimilarity(i,j)
            logging.info("WN: " + i + j +str(wnSim))
            logging.info("Spacy: " + i + j + str(spacySim))

            inp = str(i + '\n' + j)
            row = [inp, wnSim, spacySim]
            rows.append(row)

    table = texttable.Texttable()
    table.set_cols_align(["l", "r", "r"])
    table.add_rows(rows)
    print(table.draw())
