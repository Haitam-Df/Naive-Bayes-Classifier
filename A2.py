import pandas as pd
import string
import nltk
import matplotlib
import math

# -------------------------------------------------------
# Assignment 2
# Written by Haitam Daif 40007112
# For COMP 472 Section KX – Summer 2020
# --------------------------------------------------------

# ----> Task 1: Extract the data and build the model <----

file = 'hns_2018_2019.csv'
file2 = 'vocabulary.txt'
file3 = 'remove_word.txt'
file4 = 'stopwords.txt'

story_dict = {}     # dictionary containing every story word
ask_dict = {}       # dictionary containing every ask_hn word
show_dict = {}      # dictionary containing every show_hn word
poll_dict = {}      # dictionary containing every poll word

#  Read data from the csv file given
data = pd.read_csv(file, delimiter=',', skiprows=1, names=['Object ID', 'Title', 'Post Type', 'Author',
                                                           'Created At', 'URL', 'Points', 'Number of Comments', 'year'])

#  Setting training_set and testing_set for 2018 and 2019 respectively
training_set = data[(data['Created At'] > '2018-01-01') & (data['Created At'] < '2019-01-01')]
testing_set = data[(data['Created At'] > '2019-01-01')]

remove_word_txt = open("remove_word.txt", "w+", encoding="utf-8")

# output unnecessary things to the remove_word.txt file
unnecessary_words = ['the', 'to', 'of', 'in', 'a', 'and', '’', '–', '“', '”', '‘', '—']
for i in unnecessary_words:
    remove_word_txt.write(i + '\n')
for i in training_set['Title'].to_string():
    if i in string.punctuation:
        remove_word_txt.write(i + '\n')

#  Remove every punctuation in 'Title' sentences to count words properly
flatData_train = training_set['Title'].apply(lambda x: x.lower())
flatData_train = training_set['Title'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

#  Fold the 'Title' to lowercase and tokenize the data set
splitter = flatData_train.str.lower().str.cat(sep=' ')
vocabulary = nltk.tokenize.word_tokenize(splitter)  # Tokenized every word of the data set in a single list
distribution = nltk.FreqDist(vocabulary)
freq = pd.DataFrame(distribution.most_common(len(data)), columns=['Word', 'Frequency'])
words_sorted = set(vocabulary)
words_sorted_ex1 = words_sorted.copy()
words_sorted_ex2 = words_sorted.copy()
words_sorted_ex3 = words_sorted.copy()

# remove words that are not relevant to a classification
remove_word_file = open(file3, encoding="utf-8")
line = remove_word_file.read()
remove_words = line.split()
for word in remove_words:
    if word in words_sorted:
        words_sorted.remove(word)

#  Create a new csv file with only the words/sentences in the 'Title' section
flat_csv_train = freq.to_csv('vocabulary.txt', index=False)

#  Counts the total number of Post Type for each class in the training set
story_total = len(flatData_train.loc[training_set['Post Type'] == 'story'])
ask_total = len(flatData_train.loc[training_set['Post Type'] == 'ask_hn'])
show_total = len(flatData_train.loc[training_set['Post Type'] == 'show_hn'])
poll_total = len(flatData_train.loc[training_set['Post Type'] == 'poll'])

#  all 'Title' words are stripped from special characters to be entered in the corresponding dictionary
freq_story = {}     # frequency of word in each class
freq_ask = {}
freq_show = {}
freq_poll = {}
story_total_words = 0   # total count of words in each class
ask_total_words = 0
show_total_words = 0
poll_total_words = 0
w_story = ''            # frequency of wi in the training set for each class
w_ask = ''
w_show = ''
w_poll = ''
story_cp_dict = {}      # dictionary with the conditional probability of each word for each class
ask_cp_dict = {}
show_cp_dict = {}
poll_cp_dict = {}

f = open("model-2018.txt", "w+", encoding="utf-8")

# initialize all the dictionaries frequency to 0
for word in vocabulary:
    freq_story[word] = 0
    freq_ask[word] = 0
    freq_show[word] = 0
    freq_poll[word] = 0

for i, row in training_set.iterrows():  # loops through the testing set data
    post_Type = row['Post Type']  # assign post type section to the variable (story,ask,show,poll)
    line = row['Title']           # assign all the title section to the variable
    line = line.strip()           # remove useless spaces
    lowercase = str.lower(line)   # cast to lowercase
    clean_set = lowercase.translate(str.maketrans('', '', string.punctuation))  # remove useless punctuations
    tokens = nltk.tokenize.word_tokenize(clean_set)  # Tokenized each line of the data set separately

    for word in tokens:  # fill every dictionary for each 'Post Type'
        if post_Type == 'story':
            story_total_words += 1
            story_dict[word] = word
            if word in story_dict:
                freq_story[word] += 1   # frequency of a word in story
        elif post_Type == 'ask_hn':
            ask_total_words += 1
            ask_dict[word] = word
            if word in ask_dict:
                freq_ask[word] += 1     # frequency of a word in ask_hn
        elif post_Type == 'show_hn':
            show_total_words += 1
            show_dict[word] = word
            if word in show_dict:
                freq_show[word] += 1    # frequency of a word in show_hn
        elif post_Type == 'poll':
            poll_total_words += 1
            poll_dict[word] = word
            if word in poll_dict:
                freq_poll[word] += 1    # frequency of a word in poll

smoothing = 0.5
smoothing_denominator = 0.5*len(words_sorted)
for i, word in enumerate(sorted(words_sorted)):
    w_story = freq_story[word]
    w_ask = freq_ask[word]
    w_show = freq_show[word]
    w_poll = freq_poll[word]

    if story_total_words != 0:  # check if the post type is empty
        cp_story = (w_story + smoothing) / (story_total_words + smoothing_denominator)
        story_cp_dict[word] = cp_story  # fill the dictionary with the probabilities answer
    else:
        cp_story = 0  # if the post is empty than probability set to 0 right away
    if ask_total_words != 0:
        cp_ask = (w_ask + smoothing) / (ask_total_words + smoothing_denominator)
        ask_cp_dict[word] = cp_ask
    else:
        cp_ask = 0
    if show_total_words != 0:
        cp_show = (w_show + smoothing) / (show_total_words + smoothing_denominator)
        show_cp_dict[word] = cp_show
    else:
        cp_show = 0
    if poll_total_words != 0:
        cp_poll = (w_poll + smoothing) / (poll_total_words + smoothing_denominator)
        poll_cp_dict[word] = cp_poll
    else:
        cp_poll = 0

        f.write(str(i+1) + '  ' + word + '  ' + str(w_story) + '  ' + str("{:.8f}".format(float(cp_story)))
                + '  ' + str(w_ask) + '  ' + str("{:.8f}".format(float(cp_ask))) + '  ' + str(w_show) + '  '
                + str("{:.8f}".format(float(cp_show))) + '  ' + str(w_poll) + '  '
                + str("{:.8f}".format(float(cp_poll))) + '\n')
f.close()
print()
# ------------------------------------------------------
#      Task 2: Use ML Classifier to test data set
# ------------------------------------------------------

#  Remove every punctuation in 'Title' sentences to count words properly
flatData_test = testing_set['Title'].apply(lambda x: x.lower())
flatData_test = testing_set['Title'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

#  Fold the 'Title' to lowercase and tokenize the testing data set
split_test = flatData_test.str.lower().str.cat(sep=' ')
vocabulary2 = nltk.tokenize.word_tokenize(split_test)  # Tokenized every word of the data set in a single list
distribution2 = nltk.FreqDist(vocabulary2)
freq2 = pd.DataFrame(distribution2.most_common(len(data)), columns=['Word', 'Frequency'])
words_sorted2 = set(vocabulary2)


#  Counts the total number of Post Type for each class in the training set
story_total2 = len(flatData_test.loc[testing_set['Post Type'] == 'story'])
ask_total2 = len(flatData_test.loc[testing_set['Post Type'] == 'ask_hn'])
show_total2 = len(flatData_test.loc[testing_set['Post Type'] == 'show_hn'])
poll_total2 = len(flatData_test.loc[testing_set['Post Type'] == 'poll'])
#print(story_total2, "  ", show_total2, '  ', ask_total2, '  ', poll_total2)

#  Probability of each Post Type
p_story = story_total2/(story_total2+show_total2+ask_total2+poll_total2)
p_ask = ask_total2/(story_total2+show_total2+ask_total2+poll_total2)
p_show = show_total2/(story_total2+show_total2+ask_total2+poll_total2)
p_poll = poll_total2/(story_total2+show_total2+ask_total2+poll_total2)
# print(p_story, "  ", p_ask, '  ', p_show, '  ', p_poll)

f2 = open("baseline-result.txt", "w+", encoding="utf-8")

counter = 0
score_story = 0
score_ask = 0
score_show = 0
score_poll = 0

for i, row in testing_set.iterrows():
    counter += 1
    sum_score_story = 0     # will sum the scores for each line of each class in the 2019 data set
    sum_score_ask = 0
    sum_score_show = 0
    sum_score_poll = 0
    highest_score = 0       # highest score to choose which class will it be predicted to
    predicted_class = ''    # predicted class based on the highest score
    classification = ''
    correct_class = row['Post Type']
    line2 = row['Title']
    line2 = line2.strip()
    lowercase2 = str.lower(line2)
    clean_set_test = lowercase2.translate(str.maketrans('', '', string.punctuation))
    tokens2 = nltk.tokenize.word_tokenize(clean_set_test)  # Tokenized each line of the data set separately

    for word in tokens2:  # in the list of words for 2019
        if word in words_sorted:  # if a word of 2019 is also in 2018
            if p_story != 0:      # make sure we don't log10(0) since it would crash the program
                score_story = math.log10(p_story) + math.log10(story_cp_dict[word])
            else:
                score_story = 0
            if p_ask != 0:
                score_ask = math.log10(p_ask) + math.log10(ask_cp_dict[word])
            else:
                score_ask = 0
            if p_show != 0:
                score_show = math.log10(p_show) + math.log10(show_cp_dict[word])
            else:
                score_show = 0
            if p_poll != 0:
                score_poll = math.log10(p_poll) + math.log10(poll_cp_dict[word])
            else:
                score_poll = 0
            sum_score_story += score_story
            sum_score_ask += score_ask
            sum_score_show += score_show
            sum_score_poll += score_poll

    if sum_score_story != 0:    # making sure the score is not 0 which would mean the class is empty
        highest_score = sum_score_story  # setting the highest score to the score of story
        predicted_class = 'story'        # if the this ends up being the highest score the classifier will set the class
    if sum_score_ask > highest_score and sum_score_ask != 0:
        highest_score = sum_score_ask
        predicted_class = 'ask_hn'
    elif sum_score_show > highest_score and sum_score_show != 0:
        highest_score = sum_score_show
        predicted_class = 'show_hn'
    elif sum_score_poll > highest_score and sum_score_poll != 0:
        highest_score = sum_score_poll
        predicted_class = 'poll'

    if predicted_class == correct_class:  # will set if the classifier prediction is right or wrong
        classification = 'right'
    else:
        classification = 'wrong'

    f2.write(str(counter) + '  ' + str(line2) + '  ' + str(predicted_class) + '  '
             + str(sum_score_story) + '  ' + str(sum_score_ask) + '  ' +
             str(sum_score_show) + '  ' + str(sum_score_poll) + '  '
             + str(correct_class) + '  ' + str(classification) + '\n')
f2.close()

#  ----> Task 3: Experiments with the classifier <----

# -----------------------------------------------
#       Experiment 1: Stop Word Filtering
# -----------------------------------------------
freq_story = {}     # frequency of word in each class
freq_ask = {}
freq_show = {}
freq_poll = {}
story_total_words = 0   # total count of words in each class
ask_total_words = 0
show_total_words = 0
poll_total_words = 0
w_story = ''            # frequency of wi in the training set for each class
w_ask = ''
w_show = ''
w_poll = ''
story_cp_dict = {}      # dictionary with the conditional probability of each word for each class
ask_cp_dict = {}
show_cp_dict = {}
poll_cp_dict = {}

stop_word_file = open(file4, encoding="utf-8")

line = stop_word_file.read()
stop_words = line.split()
for word in stop_words:
    if word in words_sorted_ex1:
        words_sorted_ex1.remove(word)

f_stop = open("stopword-model.txt", "w+", encoding="utf-8")

# initialize all the dictionaries frequency to 0
for word in vocabulary:
    freq_story[word] = 0
    freq_ask[word] = 0
    freq_show[word] = 0
    freq_poll[word] = 0

for i, row in training_set.iterrows():  # loops through the testing set data
    post_Type = row['Post Type']  # assign post type section to the variable (story,ask,show,poll)
    line = row['Title']           # assign all the title section to the variable
    line = line.strip()           # remove useless spaces
    lowercase = str.lower(line)   # cast to lowercase
    clean_set = lowercase.translate(str.maketrans('', '', string.punctuation))  # remove useless punctuations
    tokens = nltk.tokenize.word_tokenize(clean_set)  # Tokenized each line of the data set separately

    for word in tokens:  # fill every dictionary for each 'Post Type'
        if post_Type == 'story':
            story_total_words += 1
            story_dict[word] = word
            if word in story_dict:
                freq_story[word] += 1   # frequency of a word in story
        elif post_Type == 'ask_hn':
            ask_total_words += 1
            ask_dict[word] = word
            if word in ask_dict:
                freq_ask[word] += 1     # frequency of a word in ask_hn
        elif post_Type == 'show_hn':
            show_total_words += 1
            show_dict[word] = word
            if word in show_dict:
                freq_show[word] += 1    # frequency of a word in show_hn
        elif post_Type == 'poll':
            poll_total_words += 1
            poll_dict[word] = word
            if word in poll_dict:
                freq_poll[word] += 1    # frequency of a word in poll

smoothing = 0.5
smoothing_denominator = 0.5*len(words_sorted_ex1)
for i, word in enumerate(sorted(words_sorted_ex1)):
    w_story = freq_story[word]
    w_ask = freq_ask[word]
    w_show = freq_show[word]
    w_poll = freq_poll[word]

    if story_total_words != 0:  # check if the post type is empty
        cp_story = (w_story + smoothing) / (story_total_words + smoothing_denominator)
        story_cp_dict[word] = cp_story  # fill the dictionary with the probabilities answer
    else:
        cp_story = 0  # if the post is empty than probability set to 0 right away
    if ask_total_words != 0:
        cp_ask = (w_ask + smoothing) / (ask_total_words + smoothing_denominator)
        ask_cp_dict[word] = cp_ask
    else:
        cp_ask = 0
    if show_total_words != 0:
        cp_show = (w_show + smoothing) / (show_total_words + smoothing_denominator)
        show_cp_dict[word] = cp_show
    else:
        cp_show = 0
    if poll_total_words != 0:
        cp_poll = (w_poll + smoothing) / (poll_total_words + smoothing_denominator)
        poll_cp_dict[word] = cp_poll
    else:
        cp_poll = 0

        f_stop.write(str(i+1) + '  ' + word + '  ' + str(w_story) + '  ' + str("{:.8f}".format(float(cp_story)))
                + '  ' + str(w_ask) + '  ' + str("{:.8f}".format(float(cp_ask))) + '  ' + str(w_show) + '  '
                + str("{:.8f}".format(float(cp_show))) + '  ' + str(w_poll) + '  '
                + str("{:.8f}".format(float(cp_poll))) + '\n')
f_stop.close()

f_stop_result = open("stopword-result.txt", "w+", encoding="utf-8")

counter = 0
score_story = 0
score_ask = 0
score_show = 0
score_poll = 0

for i, row in testing_set.iterrows():
    counter += 1
    sum_score_story = 0     # will sum the scores for each line of each class in the 2019 data set
    sum_score_ask = 0
    sum_score_show = 0
    sum_score_poll = 0
    highest_score = 0       # highest score to choose which class will it be predicted to
    predicted_class = ''    # predicted class based on the highest score
    classification = ''
    correct_class = row['Post Type']
    line2 = row['Title']
    line2 = line2.strip()
    lowercase2 = str.lower(line2)
    clean_set_test = lowercase2.translate(str.maketrans('', '', string.punctuation))
    tokens2 = nltk.tokenize.word_tokenize(clean_set_test)  # Tokenized each line of the data set separately

    for word in tokens2:  # in the list of words for 2019
        if word in words_sorted_ex1:  # if a word of 2019 is also in 2018
            if p_story != 0:      # make sure we don't log10(0) since it would crash the program
                score_story = math.log10(p_story) + math.log10(story_cp_dict[word])
            else:
                score_story = 0
            if p_ask != 0:
                score_ask = math.log10(p_ask) + math.log10(ask_cp_dict[word])
            else:
                score_ask = 0
            if p_show != 0:
                score_show = math.log10(p_show) + math.log10(show_cp_dict[word])
            else:
                score_show = 0
            if p_poll != 0:
                score_poll = math.log10(p_poll) + math.log10(poll_cp_dict[word])
            else:
                score_poll = 0
            sum_score_story += score_story
            sum_score_ask += score_ask
            sum_score_show += score_show
            sum_score_poll += score_poll

    if sum_score_story != 0:    # making sure the score is not 0 which would mean the class is empty
        highest_score = sum_score_story  # setting the highest score to the score of story
        predicted_class = 'story'        # if the this ends up being the highest score the classifier will set the class
    if sum_score_ask > highest_score and sum_score_ask != 0:
        highest_score = sum_score_ask
        predicted_class = 'ask_hn'
    elif sum_score_show > highest_score and sum_score_show != 0:
        highest_score = sum_score_show
        predicted_class = 'show_hn'
    elif sum_score_poll > highest_score and sum_score_poll != 0:
        highest_score = sum_score_poll
        predicted_class = 'poll'

    if predicted_class == correct_class:  # will set if the classifier prediction is right or wrong
        classification = 'right'
    else:
        classification = 'wrong'

    f_stop_result.write(str(counter) + '  ' + str(line2) + '  ' + str(predicted_class) + '  '
             + str(sum_score_story) + '  ' + str(sum_score_ask) + '  ' +
             str(sum_score_show) + '  ' + str(sum_score_poll) + '  '
             + str(correct_class) + '  ' + str(classification) + '\n')
f_stop_result.close()

# -----------------------------------------------
#       Experiment 2: Word Length Filtering
# -----------------------------------------------
freq_story = {}     # frequency of word in each class
freq_ask = {}
freq_show = {}
freq_poll = {}
story_total_words = 0   # total count of words in each class
ask_total_words = 0
show_total_words = 0
poll_total_words = 0
w_story = ''            # frequency of wi in the training set for each class
w_ask = ''
w_show = ''
w_poll = ''
story_cp_dict = {}      # dictionary with the conditional probability of each word for each class
ask_cp_dict = {}
show_cp_dict = {}
poll_cp_dict = {}

f_word = open("wordlength-model.txt", "w+", encoding="utf-8")

for word in words_sorted:
    if len(word) <= 2 or len(word) >= 9:
        words_sorted_ex2.remove(word)


# initialize all the dictionaries frequency to 0
for word in vocabulary:
    freq_story[word] = 0
    freq_ask[word] = 0
    freq_show[word] = 0
    freq_poll[word] = 0

for i, row in training_set.iterrows():  # loops through the testing set data
    post_Type = row['Post Type']  # assign post type section to the variable (story,ask,show,poll)
    line = row['Title']           # assign all the title section to the variable
    line = line.strip()           # remove useless spaces
    lowercase = str.lower(line)   # cast to lowercase
    clean_set = lowercase.translate(str.maketrans('', '', string.punctuation))  # remove useless punctuations
    tokens = nltk.tokenize.word_tokenize(clean_set)  # Tokenized each line of the data set separately

    for word in tokens:  # fill every dictionary for each 'Post Type'
        if post_Type == 'story':
            story_total_words += 1
            story_dict[word] = word
            if word in story_dict:
                freq_story[word] += 1   # frequency of a word in story
        elif post_Type == 'ask_hn':
            ask_total_words += 1
            ask_dict[word] = word
            if word in ask_dict:
                freq_ask[word] += 1     # frequency of a word in ask_hn
        elif post_Type == 'show_hn':
            show_total_words += 1
            show_dict[word] = word
            if word in show_dict:
                freq_show[word] += 1    # frequency of a word in show_hn
        elif post_Type == 'poll':
            poll_total_words += 1
            poll_dict[word] = word
            if word in poll_dict:
                freq_poll[word] += 1    # frequency of a word in poll

smoothing = 0.5
smoothing_denominator = 0.5*len(words_sorted_ex2)
for i, word in enumerate(sorted(words_sorted_ex2)):
    w_story = freq_story[word]
    w_ask = freq_ask[word]
    w_show = freq_show[word]
    w_poll = freq_poll[word]

    if story_total_words != 0:  # check if the post type is empty
        cp_story = (w_story + smoothing) / (story_total_words + smoothing_denominator)
        story_cp_dict[word] = cp_story  # fill the dictionary with the probabilities answer
    else:
        cp_story = 0  # if the post is empty than probability set to 0 right away
    if ask_total_words != 0:
        cp_ask = (w_ask + smoothing) / (ask_total_words + smoothing_denominator)
        ask_cp_dict[word] = cp_ask
    else:
        cp_ask = 0
    if show_total_words != 0:
        cp_show = (w_show + smoothing) / (show_total_words + smoothing_denominator)
        show_cp_dict[word] = cp_show
    else:
        cp_show = 0
    if poll_total_words != 0:
        cp_poll = (w_poll + smoothing) / (poll_total_words + smoothing_denominator)
        poll_cp_dict[word] = cp_poll
    else:
        cp_poll = 0

        f_word.write(str(i+1) + '  ' + word + '  ' + str(w_story) + '  ' + str("{:.8f}".format(float(cp_story)))
                + '  ' + str(w_ask) + '  ' + str("{:.8f}".format(float(cp_ask))) + '  ' + str(w_show) + '  '
                + str("{:.8f}".format(float(cp_show))) + '  ' + str(w_poll) + '  '
                + str("{:.8f}".format(float(cp_poll))) + '\n')
f_word.close()

f_word_result = open("wordlength-result.txt", "w+", encoding="utf-8")

counter = 0
score_story = 0
score_ask = 0
score_show = 0
score_poll = 0

for i, row in testing_set.iterrows():
    counter += 1
    sum_score_story = 0     # will sum the scores for each line of each class in the 2019 data set
    sum_score_ask = 0
    sum_score_show = 0
    sum_score_poll = 0
    highest_score = 0       # highest score to choose which class will it be predicted to
    predicted_class = ''    # predicted class based on the highest score
    classification = ''
    correct_class = row['Post Type']
    line2 = row['Title']
    line2 = line2.strip()
    lowercase2 = str.lower(line2)
    clean_set_test = lowercase2.translate(str.maketrans('', '', string.punctuation))
    tokens2 = nltk.tokenize.word_tokenize(clean_set_test)  # Tokenized each line of the data set separately

    for word in tokens2:  # in the list of words for 2019
        if word in words_sorted_ex2:  # if a word of 2019 is also in 2018
            if p_story != 0:      # make sure we don't log10(0) since it would crash the program
                score_story = math.log10(p_story) + math.log10(story_cp_dict[word])
            else:
                score_story = 0
            if p_ask != 0:
                score_ask = math.log10(p_ask) + math.log10(ask_cp_dict[word])
            else:
                score_ask = 0
            if p_show != 0:
                score_show = math.log10(p_show) + math.log10(show_cp_dict[word])
            else:
                score_show = 0
            if p_poll != 0:
                score_poll = math.log10(p_poll) + math.log10(poll_cp_dict[word])
            else:
                score_poll = 0
            sum_score_story += score_story
            sum_score_ask += score_ask
            sum_score_show += score_show
            sum_score_poll += score_poll

    if sum_score_story != 0:    # making sure the score is not 0 which would mean the class is empty
        highest_score = sum_score_story  # setting the highest score to the score of story
        predicted_class = 'story'        # if the this ends up being the highest score the classifier will set the class
    if sum_score_ask > highest_score and sum_score_ask != 0:
        highest_score = sum_score_ask
        predicted_class = 'ask_hn'
    elif sum_score_show > highest_score and sum_score_show != 0:
        highest_score = sum_score_show
        predicted_class = 'show_hn'
    elif sum_score_poll > highest_score and sum_score_poll != 0:
        highest_score = sum_score_poll
        predicted_class = 'poll'

    if predicted_class == correct_class:  # will set if the classifier prediction is right or wrong
        classification = 'right'
    else:
        classification = 'wrong'

    f_word_result.write(str(counter) + '  ' + str(line2) + '  ' + str(predicted_class) + '  '
             + str(sum_score_story) + '  ' + str(sum_score_ask) + '  ' +
             str(sum_score_show) + '  ' + str(sum_score_poll) + '  '
             + str(correct_class) + '  ' + str(classification) + '\n')
f_word_result.close()

# -----------------------------------------------
#     Experiment 3: Infrequent Word Filtering
# -----------------------------------------------

freq_story = {}     # frequency of word in each class
freq_ask = {}
freq_show = {}
freq_poll = {}
story_total_words = 0   # total count of words in each class
ask_total_words = 0
show_total_words = 0
poll_total_words = 0
w_story = ''            # frequency of wi in the training set for each class
w_ask = ''
w_show = ''
w_poll = ''
story_cp_dict = {}      # dictionary with the conditional probability of each word for each class
ask_cp_dict = {}
show_cp_dict = {}
poll_cp_dict = {}

f_infrequent = open("infrequent-model.txt", "w+", encoding="utf-8")

for index, row in freq.iterrows():
    if row['Frequency'] == 1:
        words_sorted_ex3.remove(row['Word'])


# initialize all the dictionaries frequency to 0
for word in vocabulary:
    freq_story[word] = 0
    freq_ask[word] = 0
    freq_show[word] = 0
    freq_poll[word] = 0

for i, row in training_set.iterrows():  # loops through the testing set data
    post_Type = row['Post Type']  # assign post type section to the variable (story,ask,show,poll)
    line = row['Title']           # assign all the title section to the variable
    line = line.strip()           # remove useless spaces
    lowercase = str.lower(line)   # cast to lowercase
    clean_set = lowercase.translate(str.maketrans('', '', string.punctuation))  # remove useless punctuations
    tokens = nltk.tokenize.word_tokenize(clean_set)  # Tokenized each line of the data set separately

    for word in tokens:  # fill every dictionary for each 'Post Type'
        if post_Type == 'story':
            story_total_words += 1
            story_dict[word] = word
            if word in story_dict:
                freq_story[word] += 1   # frequency of a word in story
        elif post_Type == 'ask_hn':
            ask_total_words += 1
            ask_dict[word] = word
            if word in ask_dict:
                freq_ask[word] += 1     # frequency of a word in ask_hn
        elif post_Type == 'show_hn':
            show_total_words += 1
            show_dict[word] = word
            if word in show_dict:
                freq_show[word] += 1    # frequency of a word in show_hn
        elif post_Type == 'poll':
            poll_total_words += 1
            poll_dict[word] = word
            if word in poll_dict:
                freq_poll[word] += 1    # frequency of a word in poll

smoothing = 0.5
smoothing_denominator = 0.5*len(words_sorted_ex3)
for i, word in enumerate(sorted(words_sorted_ex3)):
    w_story = freq_story[word]
    w_ask = freq_ask[word]
    w_show = freq_show[word]
    w_poll = freq_poll[word]

    if story_total_words != 0:  # check if the post type is empty
        cp_story = (w_story + smoothing) / (story_total_words + smoothing_denominator)
        story_cp_dict[word] = cp_story  # fill the dictionary with the probabilities answer
    else:
        cp_story = 0  # if the post is empty than probability set to 0 right away
    if ask_total_words != 0:
        cp_ask = (w_ask + smoothing) / (ask_total_words + smoothing_denominator)
        ask_cp_dict[word] = cp_ask
    else:
        cp_ask = 0
    if show_total_words != 0:
        cp_show = (w_show + smoothing) / (show_total_words + smoothing_denominator)
        show_cp_dict[word] = cp_show
    else:
        cp_show = 0
    if poll_total_words != 0:
        cp_poll = (w_poll + smoothing) / (poll_total_words + smoothing_denominator)
        poll_cp_dict[word] = cp_poll
    else:
        cp_poll = 0

        f_infrequent.write(str(i+1) + '  ' + word + '  ' + str(w_story) + '  ' + str("{:.8f}".format(float(cp_story)))
                + '  ' + str(w_ask) + '  ' + str("{:.8f}".format(float(cp_ask))) + '  ' + str(w_show) + '  '
                + str("{:.8f}".format(float(cp_show))) + '  ' + str(w_poll) + '  '
                + str("{:.8f}".format(float(cp_poll))) + '\n')
f_infrequent.close()

f_infrequent_result = open("infrequent-result.txt", "w+", encoding="utf-8")

counter = 0
score_story = 0
score_ask = 0
score_show = 0
score_poll = 0

for i, row in testing_set.iterrows():
    counter += 1
    sum_score_story = 0     # will sum the scores for each line of each class in the 2019 data set
    sum_score_ask = 0
    sum_score_show = 0
    sum_score_poll = 0
    highest_score = 0       # highest score to choose which class will it be predicted to
    predicted_class = ''    # predicted class based on the highest score
    classification = ''
    correct_class = row['Post Type']
    line2 = row['Title']
    line2 = line2.strip()
    lowercase2 = str.lower(line2)
    clean_set_test = lowercase2.translate(str.maketrans('', '', string.punctuation))
    tokens2 = nltk.tokenize.word_tokenize(clean_set_test)  # Tokenized each line of the data set separately

    for word in tokens2:  # in the list of words for 2019
        if word in words_sorted_ex3:  # if a word of 2019 is also in 2018
            if p_story != 0:      # make sure we don't log10(0) since it would crash the program
                score_story = math.log10(p_story) + math.log10(story_cp_dict[word])
            else:
                score_story = 0
            if p_ask != 0:
                score_ask = math.log10(p_ask) + math.log10(ask_cp_dict[word])
            else:
                score_ask = 0
            if p_show != 0:
                score_show = math.log10(p_show) + math.log10(show_cp_dict[word])
            else:
                score_show = 0
            if p_poll != 0:
                score_poll = math.log10(p_poll) + math.log10(poll_cp_dict[word])
            else:
                score_poll = 0
            sum_score_story += score_story
            sum_score_ask += score_ask
            sum_score_show += score_show
            sum_score_poll += score_poll

    if sum_score_story != 0:    # making sure the score is not 0 which would mean the class is empty
        highest_score = sum_score_story  # setting the highest score to the score of story
        predicted_class = 'story'        # if the this ends up being the highest score the classifier will set the class
    if sum_score_ask > highest_score and sum_score_ask != 0:
        highest_score = sum_score_ask
        predicted_class = 'ask_hn'
    elif sum_score_show > highest_score and sum_score_show != 0:
        highest_score = sum_score_show
        predicted_class = 'show_hn'
    elif sum_score_poll > highest_score and sum_score_poll != 0:
        highest_score = sum_score_poll
        predicted_class = 'poll'

    if predicted_class == correct_class:  # will set if the classifier prediction is right or wrong
        classification = 'right'
    else:
        classification = 'wrong'

    f_infrequent_result.write(str(counter) + '  ' + str(line2) + '  ' + str(predicted_class) + '  '
             + str(sum_score_story) + '  ' + str(sum_score_ask) + '  ' +
             str(sum_score_show) + '  ' + str(sum_score_poll) + '  '
             + str(correct_class) + '  ' + str(classification) + '\n')
f_infrequent_result.close()

