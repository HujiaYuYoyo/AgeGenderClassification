# AgeGenderClassification

please make sure data/ is in the same directory as all these files
- move data.zip inside of code/
- unzip data.zip

to run the model:

python OLBmain.py

what each file does:

OLBmain.py - main file, trains and tests the data

learner.py - takes in filename of the train data and creates an OLB learner object 

inputimage.py - loads image

inputdata.py - loads labels and convert to json and store in memory

test.txt - test file for you to run, taken as an input to OLBmain.py train function

traindataclass.py - organizeds the train and test dataset and samples data for each different size

OLButils.py - runs pca and kmeans and unit testing the model

OLB.py - class object of olb

OLBlearner.py - learner object where all the reinforcement learning methods are located, such as q-learning, choose_action, update_qtable,etc

utils.py - another utils file for loading image

gender.txt - ground truth labels for gender class

age.txt - ground truth labels for age class


