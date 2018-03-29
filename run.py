#!/usr/bin/env python3

import subprocess
from subprocess import Popen, PIPE

# functions
#------------------------------


def readNgramNames(names_list):
    idx_names = []
    for names in names_list:
        idxns = {}
        with open(names) as f:
            for line in f:
                line = line.strip()
                idx, name = line.split(":")
                idxns[idx] = name
        idx_names.append(idxns)
    return idx_names


def run_bash(cmd):
    print("Weka command:", cmd)
    p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    if err:
        print(err.decode('utf-8'))
        exit(1)
    output = output.decode("utf-8")
    print(output)
    return output

def do_normalized_bow():
    # information gain only meaningful for full-vectored Normalized BOW
    print("Normalized BOW features")

    idxs = None
    basecmd = "java -cp  /usr/share/java/weka/weka.jar weka.attributeSelection.InfoGainAttributeEval -s weka.attributeSelection.Ranker  -i "

    for (datafile, idxs_names) in zip(data_infgain, idxs_names_infgain):
        cmd = basecmd + datafile
        output = run_bash(cmd)
        for line in output.split("\n"):
            if line.startswith("Selected attributes"):
                idxs = line.split()[2].split(",")
                break

        print(datafile,":")
        print("Rank index name")
        for i, j in enumerate(idxs):
            print("%d/%d : %s" % (i+1, len(idxs), j), idxs_names[j])


def do_naive_bayes():
    basecmd="java -cp  /usr/share/java/weka/weka.jar  weka.classifiers.bayes.NaiveBayes -t "
    print("Need to specify both train and test")
    exit(1)
    for datafile in data_naivebayes:
        cmd = basecmd + datafile
        output = run_bash(cmd)

#------------------------------
# end of functions

# globals
#------------------------------
# read files index-name information
names_infgain = ["../files_march28/Binucleotides.names", "../files_march28/Trinucleotides.names"]
data_infgain = ["binucleotides-data.arff","trinucleotides-data.arff"]
idxs_names_infgain = readNgramNames(names_infgain)
data_naivebayes = ["./HMM_probabilities_for_each_class.arff"]
#------------------------------

# inf gain
do_normalized_bow()
do_naive_bayes()

# other stuff
