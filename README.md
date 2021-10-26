# MLCAS2021
This repo contains Team Data's submission for the MLCAS 2021 Crop Yield Prediction Challenge.

## Contributors
 - Austin Connick
 - Luis Flores Lozano
 - Travis Myers
 - Raul Manquero-Ochoa
 - Cesar Perez

## For each file in the repository 
data_graphs.ipynb -> This File is a Jupiter notebook that is use to make graphs of the data to help understand what are the most important sections <br><br>
data_model.py -> This file hold function for changing the data to be use when training the model <br><br>
model_ista429.py -> This builds multiple Decision Tree Regressors for each day of every entry of the data, it then runs a prediction for each day and then adds an average of each prediction over the 214 days the results. <br><br>
model_tree.ipynb -> This file is a jupyter notebook that also runs the model but in a more visual way. <br><br>
prediction_over_time.npy -> This file is our prediction made with the model stored a npy array (10337,) <br><br>

## Model 
For our model we went with a Decision Tree Regressors that built a tree of possible outcomes for a given input and used this to make a prediction, we then built a Decision Tree Regressor for each day in the data and made a prediction for each input based on the given condition for that day, finally took the mean of all the prediction to get one prediction over the 214 days for each entry.

## To run
To run the model all that is need is the model_ista.py and data_model.py in the same working directory as well as the data that can be found at link-> https://drive.google.com/file/d/1DoyextA0q4mxumMAhBvqZbfZriIM9A-Y/view
and then run the model_ista429.py file, the program was tested with python 3.8.5
## Competition
Competition link -> https://eval.ai/web/challenges/challenge-page/1251/overview

## Acknowledgements 
Big thank you to our TA Emmanuel Gonzalez, for meeting with us and being extremely patient with our questions.<br><br>
Thank you to the other teams in ISTA 429 we gained a lot of knowledge by looking at their code:<br><br>
Team Mirical Grow - https://github.com/DerekColombo/ISTA_429_Midterm_Miracle_Grow<br><br>
Team Cyber Crop-Bots - https://github.com/abhi-386/ACIC_2021_Midterm<br><br>
Team X - https://github.com/jake-newton/X429midterm<br><br>

