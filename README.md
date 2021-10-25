# MLCAS2021
This repo will be our submission to the MLCAS2021 Dataset_Competition

For each file in the repository 
	data_graphs.ipynb -> This File is a Jupiter notebook that is use to make graphs of the data to help understand what are the most important sections
	data_model.py -> This file hold function for changing the data to be use when training the model
	model_ista429.py -> This builds multiple Decision Tree Regressors for each day of every entry of the data, it then runs a prediction for each day and then adds an average of each prediction over the 214 days the results. 
	model_tree.ipynb -> This file is a jupyter notebook that also runs the model but in a more visual way.
	prediction_over_time.npy -> This file is our prediction made with the model stored a npy array (10337,)
To run the model all that is need is the model_ista.py and data_model.py in the same working directory as well as the data that can be found at link-> https://drive.google.com/file/d/1DoyextA0q4mxumMAhBvqZbfZriIM9A-Y/view
and then run the model_ista429.py file, the program was tested with python 3.8.5

Competition link -> https://eval.ai/web/challenges/challenge-page/1251/overview



