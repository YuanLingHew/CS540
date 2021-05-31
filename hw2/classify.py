import os
import math


# These first two functions require os operations and so are completed for you
# Completed for you
def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d + "/"
        files = os.listdir(directory + subdir)
        for f in files:
            bow = create_bow(vocab, directory + subdir + f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


# Completed for you
def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d + '/'
        files = os.listdir(directory + subdir)
        for f in files:
            with open(directory + subdir + f, 'r', encoding="utf-8") as doc:
                for word in doc:
                    word = word.strip()
                    if word not in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


# The rest of the functions need modifications ------------------------------
# Needs modifications
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here

    doc = open(filepath, encoding="utf-8")

    for word2 in doc:
        word2 = word2.strip()
        for word in vocab:
            if word2 == word and word2 not in bow:
                bow[word] = 1
            elif word2 == word and word2 in bow:
                bow[word] += 1
        if word2 not in bow and None not in bow:
            bow[None] = 1
        elif word2 not in bow and None in bow:
            bow[None] += 1

    return bow


# Needs modifications
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    # TODO: add your code here
    total_num_files = len(training_data)

    for label in label_list:
        label_num_files = 0
        for dict in training_data:
            if dict["label"] == label:
                label_num_files += 1

        logprob[label] = math.log((label_num_files + smooth) / (total_num_files + 2))

    return logprob


# Needs modifications
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}
    # TODO: add your code here

    label_total_words = 0
    none_count = 0

    # Counts total words in label
    for dicti in training_data:
        if dicti["label"] == label:
            for k in dicti["bow"]:
                label_total_words += dicti["bow"][k]

    # Counts word count of word given label
    for word in vocab:
        word_prob[word] = 0
        w_count = 0
        for dict in training_data:
            if dict["label"] == label:
                for k in dict["bow"]:
                    if k == word:
                        w_count += dict["bow"][word]

        word_prob[word] = math.log((w_count + smooth * 1) / (label_total_words + smooth * (len(vocab) + 1)))

    # Counts word count of words which are not in vocab
    for dict in training_data:
        if dict["label"] == label:
            for k in dict["bow"]:
                if k not in word_prob:
                    none_count += dict["bow"][k]

    word_prob[None] = math.log((none_count + smooth * 1) / (label_total_words + smooth * (len(vocab) + 1)))

    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here

    retval["vocabulary"] = create_vocabulary(training_directory, cutoff)
    retval["log prior"] = prior(load_training_data(retval["vocabulary"], training_directory), label_list)
    retval["log p(w|y=2016)"] = p_word_given_label(retval["vocabulary"], load_training_data(retval["vocabulary"],
                                                                                            training_directory), "2016")
    retval["log p(w|y=2020)"] = p_word_given_label(retval["vocabulary"], load_training_data(retval["vocabulary"],
                                                                                            training_directory), "2020")

    return retval


# Needs modifications
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    retval["log p(y=2020|x)"] = model["log prior"]["2020"]
    retval["log p(y=2016|x)"] = model["log prior"]["2016"]
    retval["predicted y"] = 0

    file_bow = create_bow(model["vocabulary"], filepath)

    for k in file_bow:
        retval["log p(y=2016|x)"] += model["log p(w|y=2016)"][k]*file_bow[k]

    for k in file_bow:
        retval["log p(y=2020|x)"] += model["log p(w|y=2020)"][k]*file_bow[k]

    if retval["log p(y=2016|x)"] > retval["log p(y=2020|x)"]:
        retval["predicted y"] = "2016"

    else:
        retval["predicted y"] = "2020"

    return retval

