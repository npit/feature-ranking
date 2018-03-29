#!/usr/bin/env python3

import subprocess
from subprocess import Popen, PIPE

# functions
#------------------------------


def readNgramNames(names_bigrams, names_trigrams):
    tri_idx_names, bi_idx_names = {}, {}
    with open(names_bigrams) as f:
        for line in f:
            line = line.strip()
            idx, name = line.split(":")
            bi_idx_names[idx] = name

    with open(names_trigrams) as f:
        for line in f:
            line = line.strip()
            idx, name = line.split(":")
            tri_idx_names[idx] = name
    return bi_idx_names, tri_idx_names



def do_normalized_bow():
    # information gain only meaningful for full-vectored Normalized BOW
    print("Normalized BOW features")
    bigrams = "binucleotides-data.arff"
    trigrams = "trinucleotides-data.arff"

    idxs = None
    basecmd = "java -cp  /usr/share/java/weka/weka.jar weka.attributeSelection.InfoGainAttributeEval -s weka.attributeSelection.Ranker  -i "
    bicmd = basecmd + bigrams
    tricmd = basecmd + trigrams

    # bigrams
    p = Popen(bicmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        print(err.decode('utf-8'))
        exit(1)
    output = output.decode("utf-8")
    print("Weka command:",bicmd)
    print(output)

    for line in output.split("\n"):
        #print('[',line,']',sep='')
        if line.startswith("Selected attributes"):
            idxs = line.split()[2].split(",")
            break

    print("Bigrams")
    print("Rank index name")
    for i, j in enumerate(idxs):
        print("%d/%d : %s" % (i+1, len(idxs), j), bi_idx_names[j])

    # trigrams
    p = Popen(tricmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        print(err.decode('utf-8'))
        exit(1)
    output = output.decode("utf-8")
    print("Weka command:",tricmd)
    print(output)

    for line in output.split("\n"):
        #print(line)
        if line.startswith("Selected attributes"):
            idxs = line.split()[2].split(",")
            break
    print("Trigrams")
    print("Rank index name")
    for i, j in enumerate(idxs):
        print("%d/%d : %s" % (i+1, len(idxs), j), tri_idx_names[j])


#------------------------------
# end of functions

# globals
#------------------------------
# read index-name information
names_bigrams = "../files_march28/Binucleotides.names"
names_trigrams = "../files_march28/Trinucleotides.names"
bi_idx_names, tri_idx_names = readNgramNames(names_bigrams, names_trigrams)
#------------------------------

# inf gain
do_normalized_bow()

# other stuff
