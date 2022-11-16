import random
import pickle
import configparser
import copy
import re
import sys
    
from helpers import getStats, plotDict

def makeGenomeDictFromFasta(fname):
    di = []
    lines = []
    name = ""
    for line in open(fname, "rt"):
        if line.startswith(">"):
            if name != "":
                seq = "".join(lines)
                di.append((name,seq))
                lines = []
            name = re.findall("\(.+?\) (.+?) ", line)[-1]
        else:
            lines += [line.strip()]
            seq = "".join(lines)
        di.append((name,seq))
    return di


def get_full_data(sequencefiles, class_names, data_path):
    for inputFastaFile in sequencefiles:

        #read config file)
        train, test, val = 0.8, 0.1, 0.1

        #Read input file
        di = makeGenomeDictFromFasta(inputFastaFile)

        #display some stats:
        print ("total num of genomes: ", len(di))
        max_ = len(max(di, key = lambda x: len(x[1]))[1])
        print ("max genome length:", max_)  # max sequence length
        min_ = len(min(di, key = lambda x: len(x[1]))[1])
        print ("min genome length:", min_)  

        #extract class labels and the corresponding genomes; create class - genome list
        #genomes padded with zeros
        #di2List = list(di.items())
        di2List = di
        di2new = []

        for i in range(len(di2List)):
            classId = di2List[i][0]
            
            #uncoment for main gropus only - withour recombinations
            if ('H' in classId and 'N' in classId):
                genome = di2List[i][1].ljust(max_)
                di2new.append((classId, genome))
            #classes.add(classId)

        #get some stats about class frequency
        diClasses = getStats(di2new)
        plotDict(diClasses, 'hist.png')
        print (diClasses)

        #remove underrepresented classess
        if thresholdOccurences > 0:
            print("--------removal of underrepresented classess-------")
            d = di2new
            
            for key, value in diClasses.items():
                if value < thresholdOccurences:
                    print (key, value)
                    d = [i for i in d if i[0] != key] 
            
            diClasses2 = getStats(d)
            plotDict(diClasses2, 'hist2.png')
            print (diClasses2)
            di2new = d

        #divide data into a training and testing set
        totalNum = len(di2new)                      #training genomes in total
        R = int(round(totalNum*fracTraining))       #size of the traing set (measred in genomes)

        all_ = copy.copy(di2new)
        #selected = set()
        #train = []   # a set of tuples: (name, dna_seq)
        #test = []    # a set of tuples: (name, dna_seq)
        #
        ##training set
        #n_train = 0
        #
        #while n_train < R:
        #  r1 = random.randint(0, len(di2new) - 1)
        #  selected.add(r1)
        #  train.append(di2new[r1])
        #  di2new.remove(di2new[r1])
        #  n_train +=1
        #
        ##training set
        #test = di2new
        #n_test = len(di2new)
        #
        #print ("---training samples:", n_train)
        #print ("---testing samples:", n_test)
        #
        ##save data to file
        #pickle.dump(test, open(outputTestFile, "wb"))
        #pickle.dump(train, open(outputTrainFile, "wb"))
        #pickle.dump(all_, open(outputAllDataFile, "wb"))
