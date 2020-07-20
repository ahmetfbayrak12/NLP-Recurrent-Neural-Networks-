import math
import dynet as dy
from numpy.random import choice
import json
import re
import numpy as np
import matplotlib.pyplot as plt

unique_poem_words = list()
bigram_pairs = list()
poems = list()

word2id = dict()
id2word = dict()

glove_lookup_table = dict()
vocab = dict()
vectors = list()

def read_poems(folderPath):
    """ This function is a main function for reading dataset.
    Firstly it calls pre_processing function for doing several pre-processes,
    after that it fills unique_poem_words list which includes unique words in the poem dataset,
    it fills bigram_pairs list which includes all bigram pairs in the poem dataset,
    and finally it fills word to id (word2id) and id to word (id2word) dictionaries.

    :param folderPath: folder path of folder which will be read
    :return: it does not return anything
    """
    with open(folderPath) as json_file:
        data = json.load(json_file)
        for element in data:
            poem = pre_processing(element["poem"])
            poems.append(poem)
            get_unique_words(poem)
            generate_bigrams(poem)

    # word to id and id to word dictionaries
    for i, element in enumerate(unique_poem_words):
        word2id[element] = i
    for i, element in enumerate(unique_poem_words):
        id2word[i] = element

def pre_processing(poem):
    """ This function takes poem and do several pre-processing such as:
    adding start and end token to the poem, making newlines ("\n") seperate tokens
    for model to learn when to pass a new line while generating a sentence, and
    make all words lowercase

    :param poem: Takes poem (string)
    :return: pre-processed version (string array) of input
    """
    poem = poem.replace("\n", " \n ")
    poem = "bos " + poem + " eos"
    poem = poem.lower()
    preprocessed_poem = re.split((' '), poem)

    return preprocessed_poem

def get_unique_words(poem):
    """ This function is for finding unique words in the dataset.
    It looks for unique_poem_words list and If the word is in this list
    already it does not add, otherwise it adds to the list.

    :param poem: Takes poem (string array)
    :return: does not return anything
    """
    for index in range(len(poem)):
        word = (poem[index])
        if word not in unique_poem_words:
            unique_poem_words.append(word)

def generate_bigrams(poem):
    """ This function is for creating bigrams. After that
    all bigram pairs will be stored in the bigram_pairs list

    :param poem: Takes poem (string array)
    :return: it does not return anything
    """
    for index in range(len(poem)-2+1):
        word = ' '.join(poem[index:index + 2])
        bigram_pairs.append(word)

def read_glove_vectors(folderPath):
    """ This function is for reading pre-trained glove word embeddings and
    put them into glove_lookup_table dictionary.
    Also I put them in the vocab dictionary which is word to id dictionary, and
    vector list which includes vectors. For finding a vector of word you can pass
    index of word which is the values of vocab dictionary.

    :param folderPath: folder path of folder which will be read
    :return: it does not return anything
    """
    with open(folderPath) as txt_file:
        for i, element in enumerate(txt_file):
            word = element.split()[0]
            vector = np.array(element.split()[1:], dtype=float)
            vocab[word] = i
            vectors.append(list(map(float, element.split()[1:])))
            glove_lookup_table[word] = vector

def train_model(epoch_size):
    """ This function is for training the model.

    :param epoch_size: number of epoch will be applied for training
    :return: it does not return anything just prints the loss/epoch graph and
    loss epoch log while training
    """

    # plot lists for printing the graph
    plt_epoch_number = list()
    plt_epochs = list()

    # Training part
    for epoch in range(epoch_size):
        epoch_loss = 0.0
        for bigram_pair in bigram_pairs:
            dy.renew_cg()
            first_word = (bigram_pair.split(' ')[0])
            next_word = (bigram_pair.split(' ')[1])

            try:        # If the word is in the lookup table
                vector = lookup[vocab[first_word]].value()
            except:     # If the word is not in the lookup table which is an unkown word ("unk")
                vector = lookup[vocab["unk"]].value()
            x = dy.inputVector(vector)

            # Prediction function
            yhat = (pU * (dy.tanh(pW * x + pd)) + pb)

            # Calculate loss
            loss = dy.pickneglogsoftmax(yhat, unique_poem_words.index(next_word))
            epoch_loss += loss.scalar_value()

            # Back propagation and update
            loss.backward()
            trainer.update()

        print("Epoch %d. loss = %f" % (epoch+1, epoch_loss/len(bigram_pairs)))
        plt_epoch_number.append(epoch+1)
        plt_epochs.append(epoch_loss/len(bigram_pairs))

    plt.title("Epoch=%d, GloveDimensions=%d, PoemSize=%d, HiddenLayerSize=%d" %(epoch_size, glove_size, poem_train_size, hidden_size))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 10)
    plt.xlim(0, epoch_size)
    plt.plot(plt_epoch_number, plt_epochs)
    plt.show()

    # Save the model
    #model.save("./50dim10epoch1poem.model")
    #print("Model is saved.")

def create_poem(line_number):
    """ This is a main function of generating a poem. This function starts
    with start token initially after that it generates new word according to the previous
    word. And once the generating is done, it calculates perplexity of this generated poem.

    :param line_number: it takes line number as a integer. this parameter is for
    determining how many line will be in the generating poem
    :return: it does not return anything. it prints generated poem and perplexity of it
    """
    total_probability = 0
    count = 0
    poem = ""
    input = "bos"   # start with start token initially

    # This while is for determining the line number of generating poem
    # After every while, it refreshes the input
    while count < line_number:
        generated_word, word_probability = generate_word(input)

        # If the generated word is not newline or end token, add it to the poem
        if (generated_word != "\n" and generated_word != "eos"):
            poem += generated_word + " "
            input = generated_word

        # If the generated word is a newline, pass to the new line
        elif (generated_word == "\n"):
            count += 1
            poem += "\n"  # \\n
            input = generated_word

        # If the generated word is a end token, finish the poem
        elif (generated_word == "eos"):
            input = generated_word

        # Calculate perplexity
        total_probability += math.log(word_probability, 2)
        splitted_poem = poem.split()
        poem_perplexity = calculate_perplexity(splitted_poem, total_probability)

    # Print the results
    print("\nPoem: ")
    print(poem)
    print("Poem Perplexity:")
    print(poem_perplexity)

def generate_word(word):
    """ This function is for generating a new word according to the input word.

    :param word: it takes word as a string. this word will be used for generating
    a new word and then this generated word will be fed on this function iteratively.
    :return: it returns generated word and probability of it
    """
    dy.renew_cg() # Creating a computational graphs

    try:        # If the word is in the lookup table
        vector = lookup[vocab[word]].value()
    except:     # If the word is not in the lookup table which is out-of-vocabulary use "unk" token for unkown word
        vector = lookup[vocab["unk"]].value()

    # parameters
    W = dy.parameter(pW)
    d = dy.parameter(pd)
    U = dy.parameter(pU)
    b = dy.parameter(pb)
    x = dy.inputVector(vector)

    # prediction function
    yhat = (U * (dy.tanh(W * x + d)) + b)

    # normalization
    cum_yhat = list(np.exp(yhat.npvalue()) / sum(np.exp(yhat.npvalue())))

    # choose new word according to the predictions
    next_word = (choice(unique_poem_words, p=cum_yhat))

    # do not generate "\n" token after "\n" token.
    if (next_word == word == "\n"):
        while(next_word == "\n"):
            next_word = (choice(unique_poem_words, p=cum_yhat))
    # do not generate end token after start token otherwise there will be a no poem
    if(word == "bos" and next_word == "eos"):
        while(next_word == "eos"):
            next_word = (choice(unique_poem_words, p=cum_yhat))

    word_probability = cum_yhat[word2id[next_word]]

    return next_word, word_probability

def calculate_perplexity(poem, total_probabilities):
    """ This function is for calculating perplexity of a given poem

    :param poem: takes poem as a string array
    :param total_probabilities: total probability of a given poem
    :return: perplexity of a given poem
    """
    sum_of_log = 0
    sum_of_log -= total_probabilities
    poem_size = len(poem)
    perplexity = math.pow(2, (float(sum_of_log) / poem_size))
    return perplexity

if __name__ == '__main__':

    dataset = "unim_poem1000.json"
    read_poems(dataset)

    glove_dataset = "glove.6B.50d.txt"
    read_glove_vectors(glove_dataset)

    # Parameters for training
    poem_train_size = 1000  # Number of poem size which will be used in training
    glove_size = 50     # Dimension size of glove word embedding
    hidden_size = 100   # Dimension size of hidden layer
    input_size = 50                         # dimension size of input layer
    output_size = len(unique_poem_words)    # dimension size of output layer which is vocabulary size
    epoch_size = 10


    model = dy.Model()
    lookup = model.add_lookup_parameters((len(vectors), len(vectors[0])))   # lookup table for glove word embeddings
    lookup.init_from_array(np.array(vectors))       # pre-trained glove vectors

    # Parameters of model
    pW = model.add_parameters((hidden_size, input_size))
    pd = model.add_parameters(hidden_size)
    pU = model.add_parameters((output_size, hidden_size))
    pb = model.add_parameters(output_size)

    trainer = dy.SimpleSGDTrainer(model)

    train_model(epoch_size)

    #model.populate("./50dim10epoch1poem.model")

    poem_number = int(input('Type number of poem which you want to print: '))
    for i in range(0, poem_number):
        line_number = int(input('Type number of line for {0}. poem :'.format(i+1)))
        create_poem(line_number)