# RNN-for-Language-Identification---LIGN-167-Final-Project

We attempted to classify what language a word is in by using a character level RNN. As described there, the RNN reads words as a series of characters outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e., which language the word belongs to.

## Mathematical model

We use a RNN where the recurrent unit is an LSTM, which receives the current input (a char) along with the hidden state outputted by the LSTM at the previous time step. This hidden state and the input are used to calculate the hidden state for the next time step and an output (the prediction).

In our function training, we initialize the RNN with n_letters (the size of the vocabulary, i.e., size of the aforementioned letterDict), n_hidden (the number of features the hidden state has), and n_categories (the size of the langDict, ie. the number of languages). 

We train the model on n_iters words. We choose a random word from a random language and convert each letter into a one-hot vector. We run each letter through the RNN sequentially and then calculate the loss based on the final output. The rnn calculates the next hidden state and the output, and then applies softmax on the output to calculate the probability for each category. We use the PyTorch function NLLLoss which returns the -log P(y|s), where y a language/category and s is the prediction. We then back-propagate using the .backwards() function. For each of the weights, or parameters, in the LSTM, we subtract the learning rate multiplied by the gradient of the weight. 

## Prerequisites

Python 3.0, Anaconda Distribution, a Python IDE (we used Spyder), Pytorch

## Built With

* .[Anaconda](https://www.anaconda.com/distribution/) - Essentially contained all the packages and other software we used
* .[Spyder](https://www.spyder-ide.org/) - The IDE we used

## Authors

Sneheil Saxena
Rachel Barrow

## License
For now, our work may not be used elsewhere.

## Acknowledgments

Thanks to Prof. Leon Bergen for providing us with a thorough understanding of the basics so we could work our way up to this.
