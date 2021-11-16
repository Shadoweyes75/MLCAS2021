#get os for file struct
FROM continuumio/anaconda3
#
RUN apt-get -y update && apt-get install -y python3
#RUN apt-get install -y python3
#RUN apt-get install -y python3-numpy
#1.21.
RUN pip install numpy==1.21.3
RUN pip install pandas==1.3.4

#data used by model code to mount data or to download data from cyvers
RUN mkdir -p /dataset_competition/raining
RUN mkdir -p /dataset_competition/est_inputs
#geting results back to host
RUN mkdir -p /save
#RUN apt-get install -y python3-sklearn
#RUN apt-get install -y python3-pandas 
#RUN apt-get install -y python3-numpy
#RUN apt-get install python3
ADD model_ista429.py .
#ADD data_model.py .

#RUN ls
CMD ["python","./model_ista429.py"]