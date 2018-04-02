#!/usr/bin/env python3

import pandas
import os
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
    #print(output)
    return output


def do_normalized_bow():
    # information gain only meaningful for full-vectored Normalized BOW
    print("Normalized BOW features")

    idxs = None
    basecmd = "java -cp  /usr/share/java/weka/weka.jar weka.attributeSelection.InfoGainAttributeEval  -s weka.attributeSelection.Ranker  -i "

    for (datafile, idxs_names) in zip(data_infgain, idxs_names_infgain):
        parsing_ranked = False
        num_parsed = 0
        ranked = []
        cmd = basecmd + datafile
        output = run_bash(cmd)
        print(output)
        
        print(datafile, ":")
        print("Rank index name")
        for line in output.split("\n"):
            if line.startswith("Ranked attributes:"):
                parsing_ranked = True
                continue
            if parsing_ranked:
                if num_parsed >= num_infgain:
                    break
                line = line.strip()
                if not line:
                    break
                score, idx, _, _ = line.split()
                name = idxs_names[idx]
                print("%d/%d : %s %s" % \
                      (num_parsed+1, num_infgain, idx, name))
                num_parsed +=1


def do_naive_bayes(num_instances):
    basecmd="java -cp  /usr/share/java/weka/weka.jar  weka.classifiers.bayes.NaiveBayes "
    data = []

    for dataset in data_naivebayes:
        cmd = "%s -t %s -x 2" % (basecmd, dataset) + " -classifications CSV"
        output = run_bash(cmd)
        respath = os.path.basename(dataset)+".predictions.txt"
        with open(respath, "w") as f:
            f.write(output)
        df = pandas.read_csv(respath, skiprows=3)
        df = df.sort_values(by='prediction', ascending=False)
        first_n = pandas.DataFrame.head(df, n=num_instances)
        (index, predicted, confidence) = list(first_n['inst#']), list(first_n['predicted']), list(first_n['prediction'])
        num_data = len(index)

        print("num total prediction datum")
        for i in range(num_instances):
            ind, pred, conf = index[i], predicted[i], confidence[i]
            print("%d/%d: %d | %2.4f " % (i+1, num_data, ind, conf))


#------------------------------
# end of functions

# globals
#------------------------------
# Information gain:
# files index-name information
num_infgain = 16
names_infgain = ["../files_march28/Binucleotides.names", "../files_march28/Trinucleotides.names"]
# data (norm. bow) for information gain
data_infgain = ["binucleotides-data.arff","trinucleotides-data.arff"]

# instance sorting:
# data for naive bayes (train/test)
raw_data = ''
# norm. bow (tri, bi), NGG
data_naivebayes = ["binucleotides-data.arff", "trinucleotides-data.arff", "/home/nik/work/iit/submissions/eccb18/features/NGG_feats_all.arff"]
num_instances = 100
#------------------------------

# inf gain
idxs_names_infgain = readNgramNames(names_infgain)
do_normalized_bow()
#do_naive_bayes(num_instances)

# other stuff
