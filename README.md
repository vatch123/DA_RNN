# DA_RNN

A dual stage attention LSTM network used for time series prediction.

This is an implementation of the [paper](https://arxiv.org/abs/1704.02971) for my use case
of predicting solar irradiation values. The input sequence is multivariate with various
meterological parameters.

The model code is heavily inspired by this [blogpost](https://chandlerzuo.github.io/blog/2017/11/darnn) by **Chandler Zuo**.

This code has added functionality:
* To define various training runs with different set of hyperparameters
* Keep track of all the runs during training using `tensorboard`
