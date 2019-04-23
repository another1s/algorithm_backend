# release
* main.py contain bidirectional LSTM definition
* release.py is the one that will be calling for flask micro service

# debug
* training.py is model-associated, which is used for update network model and test its performance
* tokenlizer.py is used to generate word-embeddings and save the model
* pratice.py : load saved model to perform testing
* ddddd.py is just for test attributes and parameters of some tensorflow functon

# other 
* dataset folder should contain two dataset:pubmed dataset and dbpedia dataset
* helpfunction including some pre-processing function that are used to process and formalizing data