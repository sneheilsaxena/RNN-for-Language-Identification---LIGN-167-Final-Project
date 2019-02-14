# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict
import sys, getopt

import torch
import torch.nn as nn
import random

import json

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

langDict = {} # category_lines
langDirs = [] # all_categories

all_letters = '' # alphabet
n_letters = 0 # number of letters in the alphabet

n_hidden = 128
n_categories = 0
rnn = 0
learning_rate = 0.005
lossFcn = nn.NLLLoss()
all_losses = []

# command line arguments
write_file = '' # saves the rnn model to this file
read_file = ''  # reads this rnn model


words_bool = True     # trains using words data

parse_bool = False    # parses data in this run
test_bool = False     # tests data in this run
user_bool = False     # takes in user words in this run
graph_bool = False    # displays colored language figure
save_dict_bool = False # saves the word dictionaries

train_bool = False    # whether to train the data

num_iterations = 10000
num_bool = False

''' parses the datasets into the langDict dictionary '''
def parseSentences():
    
    dataDir = "./Datasets"
    
    # gets the list of languages
    for dirname in os.listdir(dataDir):
        langDirs.append(dirname)
        
    global n_categories
    n_categories = len(langDirs)
    
    n_iters = 50000
    
    # number of lines read from each file
    if num_bool:
        n_iters = (num_iterations * 5) / n_categories
    
    letterDict = defaultdict(int)
    
    # regex to remove non alphabetic unicode characters
    pattern = re.compile('[\W0-9_]+', re.UNICODE);
    
    # parses each language directory
    for language in langDirs:
        
        print ("Reading",language,"files...")
        langDict[language] = []
        path = dataDir + '/' + language
        
        # parses each file in the language directory
        for filename in os.listdir(path):        
            file = open(path + "/" + filename, mode="r", encoding="utf-8")        
            
            print ("\tReading",filename + "...")
            count = 0

            # parses each line of the text
            for line in file:        
                count += 1
                
                if (count % (n_iters/5) == 0):
                    print ("\t\tReading line",count)                
                if (count > n_iters):
                    break
                
                # list of words in line
                #lineList = []
                
                line = line.split("\t")[1]            
                # lowercase first word of sentence
                line = line[0].lower() + line[1:]
                
                # parses each word in the line
                for word in line.split():
                
                    # only unicode alphabetic
                    cleanword = pattern.sub('', word, re.UNICODE)
                    
                    for c in cleanword:
                        letterDict[c] += 1
                    
                    if cleanword:
                        #lineList.append(cleanword)
                        langDict[language].append(cleanword)
                
                # appends line to dict
                #langDict[language].append(lineList)
    
    global n_letters
    n_letters = len(letterDict)
    
    global all_letters
    all_letters = ''.join(list(letterDict.keys()))
    
    # saves parsed data into json file
    if save_dict_bool:
        with open('langDict.txt', 'w') as fp:
            json.dump(langDict, fp)
    
        with open('letterDict.txt', 'w') as fp:
            json.dump(letterDict, fp)
    
    print ("Done parsing data.")


def parseWords():

    dataDir = "./WordDatasets"
    
    # gets the list of languages
    for dirname in os.listdir(dataDir):
        langDirs.append(dirname)
    
    global n_categories
    n_categories = len(langDirs)
    
    n_iters = 40000
    
    if num_bool:
        # number of lines read from each file
        n_iters = num_iterations
    
    letterDict = defaultdict(int)
    
    # parses each language directory
    for language in langDirs:
        
        print ("Reading",language,"file...")
        langDict[language] = []
        path = dataDir + '/' + language
        
        # parses each file in the language directory
        for filename in os.listdir(path):        
            file = open(path + "/" + filename, mode="r", encoding="utf-8")        
            
            count = 0

            # parses each line of the text
            for line in file:
                
                word = line.partition(' ')[0]
                
                count += 1
                
                if (count % (n_iters/5) == 0):
                    print ("\tReading line",count)                
                if (count > n_iters):
                    break
                    
                for c in word:
                    letterDict[c] += 1
                    
                if word:
                    langDict[language].append(word)
    
    global n_letters
    n_letters = len(letterDict)
    
    global all_letters
    all_letters = ''.join(list(letterDict.keys()))
    
    if save_dict_bool:
        with open('langDict.txt', 'w') as fp:
            json.dump(langDict, fp)
        
        with open('letterDict.txt', 'w') as fp:
            json.dump(letterDict, fp)
    
    print ("Done parsing data.")



#---------------------------------------------------------------------------


def wordToTensor(word):
    tensor = torch.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

#-----------------------------------------------------------------------------

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)
    
    # lstm function, calculates hidden layer and output
    def forward(self, input, hidden):
        # combines input character and hidden state
        combined = torch.cat((input,hidden), 1)
        
        # calculates new hidden state
        hidden = self.i2h(combined)
        
        # calculates and normalizes output - prediction
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, hidden

    # initialiazes hidden state to zeros
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    

#----------------------Classifier---------------------------------------------------

# Classification based on RNN output
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return langDirs[category_i], category_i


#-----------------Training Helper Functions-------------------------------------
    
def randomChoice(list_):
    return list_[random.randint(0, len(list_) - 1)]

def randomTrainingExample():
    category = randomChoice(langDirs)
    word = randomChoice(langDict[category])
    category_tensor = torch.tensor([langDirs.index(category)], dtype=torch.long)
    word_tensor = wordToTensor(word)
    return category, word, category_tensor, word_tensor

def randomTrainingExample_s(lang, i):
    category = lang
    line = langDict[category][i]
    category_tensor = torch.tensor([i], dtype=torch.long)
    line_tensor = wordToTensor(line)
    return category, line, category_tensor, line_tensor

# trains a single word
def train_word (category_tensor, word_tensor):
    hidden = rnn.initHidden()
    
    rnn.zero_grad()
    
    # trains on each letter of the word
    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)
        
    # calculates loss
    loss = lossFcn(output, category_tensor)
    
    # calculates gradients - back-propogates
    loss.backward()
    
    # updates weights based on gradient descent
    for weight in rnn.parameters():
        weight.data.add_(-learning_rate, weight.grad.data)
        
    return output, loss.item()

def timeSince(since):
    now = time.time()
    secs = now - since
    mins = math.floor(secs / 60)
    secs -= mins * 60
    return '%dm %ds' % (mins, secs)

#---------------------------Training Functions-------------------------------------

def training():
    n_iters = 30000
    
    if num_bool:
        n_iters = num_iterations
    print_every = n_iters / 10
    plot_every = n_iters / 50
    
    current_loss = 0
    
    # n_hidden = features of hidden 
    global rnn
    rnn = RNN(n_letters, n_hidden, n_categories)
    
    start = time.time()
    
    # trains on n_iters words
    for i in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train_word(category_tensor, line_tensor)
        current_loss += loss
        
        if i % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = 'correct' if guess == category else 'incorrect (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (i, i / n_iters * 100, timeSince(start), loss, line, guess, correct))
    
        if i % plot_every == 0:
            global all_losses
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    
    torch.save(rnn, write_file) 

# displays a graph of the loss after training is done
def plotLoss():
    plt.figure()
    global all_losses
    plt.plot(all_losses)
    plt.show()

#------------------------------Test/Evaluate-----------------------------------------------
    
def test(): 
    
    n_tests = 50000
    if num_bool:
        n_tests = num_iterations
    print_every = n_tests / 20
    
    num_wrong = 0
    
    for i in range(n_tests):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = predict(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        
        correct_bool = guess == category
        
        if i % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = 'correct!' if guess == category else 'incorrect :( (%s)' % category
            print('%d %d%% %s / %s, %s' % (i, i / n_tests * 100, line, guess, correct))
        
        if not correct_bool:
            num_wrong += 1
       
    error = num_wrong/n_tests
        
    print ("Number of wrong guesses =",num_wrong,"out of",str(n_tests) + ". Error =",error)

def predict(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):    
        output, hidden = rnn(line_tensor[i], hidden)
    return output    


#----------------------------------Graph---------------------------------------

# displays the graph
def predictionFigure():
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 50000
    if num_bool:
        n_confusion = num_iterations
    
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = predict(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = langDirs.index(category)
        confusion[category_i][guess_i] += 1
    
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticklabels([''] + langDirs, rotation=90)
    ax.set_yticklabels([''] + langDirs)
    
    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # sphinx_gallery_thumbnail_number = 2
    plt.show()

#---------------------------User input-----------------------------------------

# returns the top 2 language choices
def top_two(input_line):
    n_predictions = 3
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = predict(wordToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, langDirs[category_index]))
            predictions.append([value, langDirs[category_index]])


def userInput():
    while True:
        name = input("Enter a word: ")
        if name == '':
            break
        top_two(name)

#---------------------------Driver-------------------------------------------

def main():
    global read_file
    global user_bool
    
    if len(sys.argv) == 1:
        print ("\npython project.py --help --parse <out_file> --test <in_file> --train <in_file>",
                   "--userInput <in_file> --graph <in_file>\n" +
                   "<infile> and <outfile> are .pt files that contain the RNN model" +
                   "\t-h (--help): prints this message\n" +
                   
                   "\t-p (--parse): trains the RNN and saves the model to out_file\n" +
                   
                   "\t-t (--test): tests the accuracy of the RNN model in in_file\n" + 
                   
                   "\t-u (--user): predicts words' languages using the RNN model in in_file\n" +
                   
                   "\t-g (--graph): graphs the accuracy of the predictions from RNN model in in_file\n" +
                   "\n\t-n (--number) : number of iterations to train\n")
        return 0
    
    options, remainder = getopt.getopt(sys.argv[1:], 'hp:t:u:g:n:',
                                     ["parse=","train=","test=","userInput=",
                                      "graph=","help","number=", "save"])
    
    # parses command line arguments
    for opt, arg in options:
        if opt in ('-h', '--help'):
            print ("\npython project.py -h -s -p <out_file> -t <in_file>",
                   "-u <in_file> -g <in_file>\n" +
                   "\t-h (--help): prints this message\n" +
                   
                   "\t-p (--parse): trains the RNN and saves the model to out_file\n" +
                   
                   "\t-t (--test): tests the accuracy of the RNN model in in_file\n" + 
                   
                   "\t-u (--user): predicts words' languages using the RNN model in in_file\n" +
                   
                   "\t-g (--graph): graphs the accuracy of the predictions from RNN model in in_file\n" +
                   "\n\t-n (--number) : number of iterations to train\n")
            return 0
        
        # parses the data
        elif opt in ('-p', '--parse'):
            if (arg == "sentence" or arg == "s"):
                global words_bool
                words_bool = False
            global parse_bool
            parse_bool = True
            
        # trains RNN and saves it to writeFile in this run
        elif opt in '--train':
            global write_file
            write_file = arg
            global train_bool
            train_bool = True
            
        elif opt in ('-n', '--number'):
            global num_bool
            num_bool = True
            global num_iterations
            num_iterations = int(arg)
            
        # tests RNN that was stored in readFile in this run
        elif opt in ('-t', '--test'):
            read_file = arg
            global test_bool 
            test_bool = True
            
        # takes in user input and outputs predictions in this run
        elif opt in ('-u', '--userInput'):
            read_file = arg
            user_bool = True
            
        elif opt in ('-g', '--graph'):
            
            read_file = arg
            global graph_bool
            graph_bool = True
            
        elif opt == '--save':
            global save_dict_bool
            save_dict_bool = True



  
    # loads dictionaries from files
    if not parse_bool and not train_bool:
        global rnn
        rnn = torch.load(read_file)
        
    if not parse_bool:
        with open("langDict.txt") as ldFile:
            global langDict
            langDict = json.loads(ldFile.read())
            
        with open("letterDict.txt") as letterFile:
            letterDict = json.loads(letterFile.read())
        
        global all_letters
        all_letters = ''.join(list(letterDict.keys()))
        global n_letters
        n_letters = len(all_letters)
        
        global langDirs
        langDirs = list(langDict.keys())
        
        global n_categories
        n_categories = len(langDirs)
        
    
    if (parse_bool):     
        # parses words from files
        if (words_bool):
            parseWords()
        else:
            parseSentences()    
       
        
    elif (train_bool):
        # trains the RNN and saves it
        training()    
        # displays loss function graph
        plotLoss() 
            
    elif (user_bool):
        userInput()
        
    elif (test_bool):
        test()
        
    elif (graph_bool):
        predictionFigure()
        
    return 0
        
    
if __name__ == "__main__": main()