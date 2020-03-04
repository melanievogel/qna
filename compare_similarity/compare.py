import spacy
import logging
import texttable
from pathlib import Path
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_md") 
nlp_bert = spacy.load("en_trf_bertbaseuncased_lg")
nlp_xlnet = spacy.load("en_trf_xlnetbasecased_lg")
nlp_roberta = spacy.load("en_trf_robertabase_lg")
#nlp = spacy.load("en_trf_distilbertbaseuncased_lg")

logging.basicConfig(filename='logs.log',level=logging.INFO)

def getWordnetSynset(word):
    synset = wn.synsets(word.rstrip())
    if synset is None or len(synset) <= 0:
        return None
    else:
        return synset[0]
    
def calcSpacySimilarity(w1,w2):
    token1 = nlp(w1)
    token2 = nlp(w2)
    return token1.similarity(token2)

def calcBertSimilarity(w1,w2):
    token1 = nlp_bert(w1)
    token2 = nlp_bert(w2)
    return token1.similarity(token2)

def calcRobertaSimilarity(w1,w2):
    token1 = nlp_roberta(w1)
    token2 = nlp_roberta(w2)
    return token1.similarity(token2)

def calcXlnetSimilarity(w1,w2):
    token1 = nlp_xlnet(w1)
    token2 = nlp_xlnet(w2)
    return token1.similarity(token2)

if __name__ == "__main__":

    with open(Path("osm_tags_sm.txt"), 'r') as f:
        osm_tag_lines = f.readlines()
    osm_tags = [i.rstrip('\n') for i in osm_tag_lines]

    search_words = ["This place offers great tea, cupcakes, and coffee."]

    rows = [['Words','Spacy (Word2Vec)','BERT', 'RoBERTa', 'XLNet']]

    result_dict = dict()    

    for i in search_words:
        # isZero = False
        # syn1 = getWordnetSynset(i)
        # if syn1 is None:
        #     isZero = True
        for j in osm_tags:
            # syn2 = getWordnetSynset(j)
            # if (syn2 is None) or isZero:
            #     wnSim = 0
            # else:
            #     wnSim = syn1.path_similarity(syn2)
            wnSim = calcSpacySimilarity(i,j)
            bertSim = calcBertSimilarity(i,j)
            robertaSim = calcRobertaSimilarity(i,j)
            xlnetSim = calcXlnetSimilarity(i,j)
            # bertSim = 0
            # robertaSim = 0

            logging.info("Similarity between "+i+" & "+j+": Spacy="+str(wnSim) + " BERT="+str(bertSim) + " RoBERTa="+str(robertaSim) + " XLNet="+str(xlnetSim))

            inp = str(i + '\n' + j)
            row = [inp, wnSim, bertSim, robertaSim, xlnetSim]
            result_dict[(i,j)] = (wnSim, bertSim, robertaSim, xlnetSim) 
            rows.append(row)

    table = texttable.Texttable()
    table.set_cols_align(["l", "r", "r", "r", "r"])
    table.add_rows(rows)
    print(table.draw())

    print("\nSorted list based on Spacy similarity:")
    sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1][0], reverse=True)[:5])
    print(sorted_dict)

    print("\nSorted list based on BERT similarity:")
    sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1][1], reverse=True)[:5])
    print(sorted_dict)

    print("\nSorted list based on RoBERTa similarity:")
    sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1][2], reverse=True)[:5])
    print(sorted_dict)

    print("\nSorted list based on XLNet similarity:")
    sorted_dict = dict(sorted(result_dict.items(), key=lambda item: item[1][3], reverse=True)[:5])
    print(sorted_dict)
