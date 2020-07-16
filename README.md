# attention_rnn
A little project made for recruitment

### evaluate.py
Demo script. A trained network weights file is needed to run. 

### train.py
The file with the training function.

### data_utils.py
Some utility functions to ease data preparation.

### model.py
The model definition.

## Training results
![Training report](https://github.com/Walusus/attention_rnn/blob/master/plots/train_report.png "Training report")
![Confusion matrix](https://github.com/Walusus/attention_rnn/blob/master/plots/conf_mat.png "Confusion matrix")
Class labels:
* 0 - AddToPlaylist
* 1 - BookRestaurant
* 2 - GetWeather
* 3 - PlayMusic
* 4 - RateBook
* 5 - SearchCreativeWork
* 6 - SearchScreeningEvent


[**Dataset source**](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines)
