#!/usr/bin/env python3

import subprocess
from subprocess import Popen, PIPE
import pandas

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


def do_naive_bayes(num_instances):
    basecmd="java -cp  /usr/share/java/weka/weka.jar  weka.classifiers.bayes.NaiveBayes "
    data = []

    for (train, test) in data_naivebayes:
        cmd = "%s -t %s -T %s" % (basecmd, train, test) + " -classifications CSV"
        output = run_bash(cmd)
        with open("temp.txt", "w") as f:
            f.write(output)
        df = pandas.read_csv("temp.txt", skiprows=3)
        df = df.sort_values(by='prediction', ascending=False)
        first_n = pandas.DataFrame.head(df, n=num_instances)
        (index, predicted, confidence) = list(first_n['inst#']), list(first_n['predicted']), list(first_n['prediction'])


        if not raw_data:
            exit(1)
        with open(raw_data) as f:
            for line in f:
                data.append(line.strip())
        print("num total prediction datum")
        for i in range(num_instances):
            ind, pred, conf = index[i], predicted[i], confidence[i]
            print("%d/%d: 2.4f |  %s" % (ind, len(data), data[ind]))


#------------------------------
# end of functions

# globals
#------------------------------
# Information gain:
# files index-name information
names_infgain = ["../files_march28/Binucleotides.names", "../files_march28/Trinucleotides.names"]
# data for information gain
data_infgain = ["binucleotides-data.arff","trinucleotides-data.arff"]
# instance sorting:
# data for naive bayes (train/test)
raw_data = ''
data_naivebayes = [("./HMM_probabilities_for_each_class.arff", "./HMM_probabilities_for_each_class.arff")]
num_instances = 100
#------------------------------

# inf gain
idxs_names_infgain = readNgramNames(names_infgain)
#do_normalized_bow()
do_naive_bayes(num_instances)

# other stuff
