import sys
import os
import shutil


arado160 = os.getcwd() + "/arado/arado_160/"
arado160test = os.getcwd() + '/arado/arado_160_test/'
arado160train = os.getcwd() + '/arado/arado_160_train/'
arado160directories = next(os.walk('arado/arado_160'))[1]


def create_files(dir):
    filesInDir = os.listdir(arado160 + '/' + dir)
    for i in xrange(0, len(filesInDir)/2):
        if(not os.path.isfile(arado160test + dir + '/' + filesInDir[i])):
            print("Copy file " + filesInDir[i] +
                  " to dir: " + arado160test + dir)
            shutil.copyfile(arado160 + dir + '/' +
                            filesInDir[i], arado160test + dir + '/' + filesInDir[i])
    for i in xrange(len(filesInDir)/2, len(filesInDir)):
        if(not os.path.isfile(arado160train + dir + '/' + filesInDir[i])):
            print("Copy file " + filesInDir[i] +
                  " to dir: " + arado160train + dir)
            shutil.copyfile(arado160 + dir + '/' +
                            filesInDir[i], arado160train + dir + '/' + filesInDir[i])


def create_directories(dir):
    dirToAddTest = arado160test + dir
    dirToAddTrain = arado160train + dir
    if not os.path.isdir(dirToAddTest):
        print('Creating directory ' + dirToAddTest)
        os.makedirs(dirToAddTest)
    if not os.path.isdir(dirToAddTrain):
        print('Creating directory ' + dirToAddTrain)
        os.makedirs(dirToAddTrain)


for dir in arado160directories:
    create_directories(dir)
    create_files(dir)
