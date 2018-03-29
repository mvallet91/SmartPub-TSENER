'''
@author: mesbahs
'''
"""
This script is used to train the NER model
"""
import subprocess
import re
import config as cfg


# Testing the results of the trained NER model on the testfile
def test(numberOfSeeds, name, numberOfIteration):
    for iteration in range(0, 2):
        outputfile = open(cfg.ROOTPATH + '/crf_trained_files/temp' + numberOfIteration + name + 'testB.txt', 'a')
        command = ('java -cp ' + cfg.ROOTPATH + '/stanford_files/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ' + 
                    cfg.ROOTPATH + '/crf_trained_files/' + name + '_text_iteration' + numberOfIteration + '_splitted' + 
                    str(numberOfSeeds) + '_' + str(iteration) + '.ser.gz -testFile ' + cfg.ROOTPATH + '/data/testB_dataset.txt')
        
        p = subprocess.call(command,
                            stdout=outputfile,
                            stderr=subprocess.STDOUT, shell=True)
        outputfile.close()

###############
# Training the Stanford NER model
def train(numberOfSeeds, name, numberOfIteration):
    for iteration in range(0, 2):
        outputfile = open(cfg.ROOTPATH + '/crf_trained_files/temp' + numberOfIteration + name + 'testA.txt', 'a')
        command = ('java -cp ' + cfg.ROOTPATH + '/stanford_files/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop ' +
                    cfg.ROOTPATH + '/prop_files/austen' + str(numberOfSeeds) + '_' + str(iteration) + '.prop')
        p = subprocess.call(command,
                            stdout=outputfile,
                            stderr=subprocess.STDOUT, shell=True)


# Generating the property file for training the Stanfor NER model
def create_austenprop(numberOfSeeds, name, numberOfIteration):
    for iteration in range(0, 2):
        outputfile = open(cfg.ROOTPATH + '/data/austen.prop', 'r')
        text = outputfile.read()
        print(text)
        modifiedpath = ('trainFile=' + cfg.ROOTPATH + '/evaluation_files_prot/' + name + '_text_iteration' + numberOfIteration + 
                        '_splitted' + str(numberOfSeeds) + '_' + str(iteration) + '.txt')
        modifiedpathtest = ('testFile=' + cfg.ROOTPATH + '/evaluation_files_prot/' + name + '_text_iteration' + numberOfIteration +
                            'test_splitted' + str(numberOfSeeds) + '_' + str(iteration) + '.txt')
        serializeTo = ('serializeTo=' + cfg.ROOTPATH + '/crf_trained_files/' + name + '_text_iteration' + numberOfIteration + 
                        '_splitted' + str(numberOfSeeds) + '_' + str(iteration) + '.ser.gz')
        edited = re.sub(r'trainFile.*?txt', modifiedpath, text, flags=re.DOTALL)
        edited = re.sub(r'testFile.*?txt', modifiedpathtest, edited, flags=re.DOTALL)
        edited = re.sub(r'serializeTo.*?gz', serializeTo, edited, flags=re.DOTALL)
        print('Edited:')
        print(edited)
        text_file = open(cfg.ROOTPATH + '/prop_files/austen' + str(numberOfSeeds) + '_' + str(iteration) + '.prop', 'w')
        text_file.write(edited)
        text_file.close()