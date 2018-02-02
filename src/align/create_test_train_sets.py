import sys
import os
import shutil


arado_160 = os.getcwd() + "/arado/arado_160/"
arado_160_test = os.getcwd() + '/arado/arado_160_test/'
arado_160_train = os.getcwd() + '/arado/arado_160_train/'
arado_160_directories = next(os.walk('arado/arado_160'))[1]


def create_files(dir):
    files_in_dir = os.listdir(arado_160 + '/' + dir)
    proportion_test_train = 2
    if(len(files_in_dir) > 2):
        proportion_test_train = 3

    for i in xrange(0, len(files_in_dir)/proportion_test_train):
        if(not os.path.isfile(arado_160_test + dir + '/' + files_in_dir[i])):
            print("Copy file " + files_in_dir[i] +
                  " to dir: " + arado_160_test + dir)
            shutil.copyfile(arado_160 + dir + '/' +
                            files_in_dir[i], arado_160_test + dir + '/' + files_in_dir[i])

    for i in xrange(len(files_in_dir)/proportion_test_train, len(files_in_dir)):
        if(not os.path.isfile(arado_160_train + dir + '/' + files_in_dir[i])):
            print("Copy file " + files_in_dir[i] +
                  " to dir: " + arado_160_train + dir)
            shutil.copyfile(arado_160 + dir + '/' +
                            files_in_dir[i], arado_160_train + dir + '/' + files_in_dir[i])


def create_directories(dir):
    dir_to_be_added_to_test = arado_160_test + dir
    dir_to_be_added_to_train = arado_160_train + dir

    if not os.path.isdir(dir_to_be_added_to_test):
        print('Creating directory ' + dir_to_be_added_to_test)
        os.makedirs(dir_to_be_added_to_test)

    if not os.path.isdir(dir_to_be_added_to_train):
        print('Creating directory ' + dir_to_be_added_to_train)
        os.makedirs(dir_to_be_added_to_train)


for dir in arado_160_directories:
    create_directories(dir)
    create_files(dir)
