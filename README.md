# DLNLP-Project
Twitter sentiment analysis with a stacked LSTM

### This project uses Tensorflow 1 and is no longer idiomatic, or indeed runnable in current form with up-to-date libraries. This may or may not be fixed in the future.

## Structure

Upon request, some clarification of the structure of this project is in order. `embeddings/` and `tweets/` contain sample data, the former being a small pretrained embedding model and the latter being a set of tweets. The embedding model was copied from elsewhere, but unfortunately its origin is no longer clear. It has 300 dimensions and roughly 16k words and is in word2vec binary format. The tweets were manually annotated as part of a project by the SFS at the University of Tübingen and each fall into exactly one of six classes.

The `config.py` file simply stores hyperparameters for the model. It is done this way instead of with some type of markup in order to allow easier importation into the model with less boilerplate clutter. Incidentally, `numberer.py` is nothing but boilerplate necessitated by the way embedding lookup layers worked in Tensorflow 1.

`model.py` is where the magic happens. This is where the model structure is defined in terms of a `Model` class. This allows us to move handles to our model around and makes the actual main function more easily extensible. It does not interfere with or work around Tensorflow variable scope. It doesn't need to.

A `Model` is instantiated, trained, and immediately discarded in `train.py`. The purpose of this project was not to actually do classification per se, but rather to implement an LSTM-based sentiment classifier in Tensorflow and tune it such that it could produce useful results; it was a class assignment, not a product. However, with the way it is designed, the model could easily be saved and/or input could be piped into it.
