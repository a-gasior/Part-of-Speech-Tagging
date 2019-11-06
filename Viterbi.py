#Andrew Gasiorowski
#Project 2

import sys, re #used for regex and reading in command line arguments
from itertools import product #used to generate all possible pos bi-grams
import operator #used for returning max value from dictionaries

#This function takes a file name and reads in the text as a string. <SOS>/<SOS> is used to represent the start of a new sentence. The function first manually prepends a SOS tag to the beginning of the first line. Then re.sub is used to replace all newline charecters with the SOS tag. Next, re.findall is used to return tuples of the form (word, pos_tag) for every word/pos in the corpus. This ordered list of tuples is returned.
def extract_tuples_train(file_name):
    with open(file_name) as file:
        #I'm making the assumption that every line is only one sentence. Therefore, I only need to insert one "start of sentence" tag per newline in file.
        corpus = '<SOS>/<SOS> ' + file.read()
        corpus = re.sub(r'\n', '<SOS>/<SOS> ', corpus)
        #There are a lot of edge cases causing problems. I will list the cases and how I handled them:
        #waterworks/NN|NNS ---> reduced to waterworks/NN
        #pianist\/bassoonist\/composer/NN ---> reduced to composer/NN
        #low*/JJ-tech/NN ---> This is an error with the input file. The professor permitted me to fix this in the input file
        #na/TO/NP ----> reduced to na/TO
        #Boy/NP.../: ---->
        #'/''.../: ---->
        #([^/ ]+)(?<!\\|\*)/([^/| ]+)(?<!:)
        corpus = re.findall(r'([^/ ]+)(?<!\\|\*)/([^/| ]+)', corpus)
        return corpus
#I'll try and explain my regex. It returns an in order list of tuples of the form (word, pos_tag) I use the word delimiter to describe the foward slash between the word and pos. It has two capture groups. ([^/ ]+) is saying, match any char that is not a delimiter or a space. Continue matching chars until you reach a delimiter. (?<!\\|\*) says accept those chars as a capture group as long as the previous char is not an escape charecter, pipe, or astrik. This is what lets us capture composer out of pianist\/bassoonist\/composer/NN. I acknowledge that I'm loosing information by not including the other two options but I accept this as okay. /([^/| ]+) says, after the delimiter, accept any char that is not the delimiter or pipe. This becomes the second capture group. This allows us to capture the single pos tag NN from waterworks/NN|NNS. Once again we are loosing information by not capturing NNS. In general my motivation for reducing multi-part tags and words is that they will not appear frequently and we stand to gain more 'meaningful' information by picking just one word and one tag.

#This function takes a list of word,bi-gram tuples. It iterates through the list 1 time and creates the three dictionaries neccessary for computing the quantities of the viterbi algorithm. I won't explain how it works at a high level but there is a line by line description.
def create_dictionaries(corpus_tuple):
    word_dict = {}
    #word_dict = {key=word : value=nested_dictionary}
    #nested_dictionary = {key=posN : value=number of times word appears as posN  ,  key=total : value=total count of word}
    pos_pos_dict = {}
    #pos_pos_dict = {key=(pos1, pos2) : value=count for this pos bi-gram}
    pos_dict = {}
    #pos_dict = {key=posN : value=count for this part of speech}
    first_iteration = True
    #We will start counting bi_grams on the second iteration
    #the following examples show some common calls and what they return:
    #tup[0] -> word
    #tup[1] -> part-of-speech
    #word_dict[tup[0]] -> {pos1:#, pos2:#, ... , posN:#}
    for index,tup in enumerate(corpus_tuple):
    #for every tupple(word) in the corpus
    #The following section creates a dictionary of the form word:dictionary{total:#, pos1:#, pos2:#, ... , posN:#} where total refers to the total count for that word (independant to tag) and posN refers to the count for the word as tag N
        if tup[0] not in word_dict:
            #if the word has not been seen yet
            word_dict[tup[0]] = {tup[1]:1}
            #Word becomes key. Value is nested dict. Nested dict receives the pos tag with an initial count of zero
        else:
            #if the word has already been seen
            if tup[1] not in word_dict[tup[0]]:
                #if the word's pos is not in the nested dict for the word
                word_dict[tup[0]][tup[1]] = 1
                    #word_dict[tup[0]]['total'] = word_dict[tup[0]]['total'] + 1
                #add the word's pos to nested dict with count 1. incrament total count
            else:
                #if the word's pos is already in the nested dict for the word
                word_dict[tup[0]][tup[1]] = word_dict[tup[0]][tup[1]] + 1
                    #word_dict[tup[0]]['total'] = word_dict[tup[0]]['total'] + 1
                #incrament the pos tag by one. incrament total occurences of the word by 1.
    ###The next section creates the pos bi-gram counts for the corpus.
        if first_iteration == False:
            #This section is skipped on the first iteration to avoid index error
            pos_tup = (corpus_tuple[index-1][1],tup[1])
            #tuple is (pos1, pos2) -> it's the bigram representations for parts of speech
            if pos_tup not in pos_pos_dict:
                #if we have not seen this pos bi-gram yet
                pos_pos_dict[pos_tup] = 1
            else:
                #if we have already seen this pos bi-gram
                pos_pos_dict[pos_tup] = pos_pos_dict[pos_tup] + 1
    ###The next section creates the raw counts for occurences of parts of speech independent to words
        if tup[1] not in pos_dict:
            #if we have not seen this part of speech yet
            pos_dict[tup[1]] = 1
        else:
            #if we have alrady seen this part of speech tag
            pos_dict[tup[1]] = pos_dict[tup[1]] + 1
        first_iteration = False
    return word_dict, pos_pos_dict, pos_dict

#This function takes a test file name as a parameter and returns two lists. corpus_list is a list of lists. Eache nested list represents a sentence composed of words. actual pos list is the coresponding tags. It broadly mirrors extract_tuples_train except it doesn't inset <SOS> tags at the beginning of strings since they are unneccessary via my viterbi implementation.
def extract_tuples_test(file_name):
    corpus_list = []#This holds the sentences in the testing corpus
    actual_pos_list = []#This hold the actual pos tag
    with open(file_name) as file:
        for line in file:
            sentence_list = re.findall(r'([^/ ]+)(?<!\\|\*)/([^/| ]+)', line)
            sentence_list = list(zip(*sentence_list))
            corpus_list.append(sentence_list[0])
            actual_pos_list.append(sentence_list[1])
    return corpus_list, actual_pos_list

#This is the viterbi algorithm for predicted part of speech tags. It takes the three dictionaries that represent the training set and corpus_list. corpus_list is a list of lists that represents the testing data after tags have been stripped
def viterbi_algorithm(word_dict, pos_bi_gram_dict, pos_dict, corpus_list):
    predicted_tags = []
    for sentence in corpus_list:
        #Initialization Step
        score = []#score[n] holds a list of probabilities for word n.
        #score[n][i] holds the prob for word n index i
        score.append([])
        tag_list = []#tag_list[n] holds a list of tags for word n
        #tag_list[n][i] holds the tag for word n index i
        tag_list.append([])
        back_ptr = []
        back_ptr.append(0)
        try:#if we've seen the word before
            tag_list[0] = list(word_dict[sentence[0]].keys())
            #We will calulate all tags
        except KeyError:#if we haven't seen the word before
            if sentence[0].lower() in word_dict:
                #if we've seen this word when the first char is lowercase
                tag_list[0] = list(word_dict[sentence[0].lower()].keys())
                #assign this word the same tag that it gets when the first char is lowercase
                word_dict[sentence[0]] = word_dict[sentence[0].lower()]
            else:#if we've really never seen this word before
                tag_list[0] = ['NN']
                #We will say it is a noun
                word_dict[sentence[0]] = {'NN':1}
                #We will add the word to our corpus as a noun, that we've seen once, to avoid future Key Errors 
        for tag in tag_list[0]:
            #for every tag that this word has
            prob_of_tag = (word_dict[sentence[0]][tag]/pos_dict[tag]) * (pos_bi_gram_dict[('<SOS>',tag)]/pos_dict['<SOS>'])
            #Calc probability of first word being tag
            score[0].append(prob_of_tag)
            #score[n] holds a list of probabilities
        #Iteration Step
        for idx,word in enumerate(sentence[1:]):
            this_word_prob_list = []
            tag_list.append([])
            idx = idx + 1#Since we're starting at second word we must incrament the counter
            max_back_ptr = []#Every current tag will have a back pointer to the previous word's tag that maximizes the current's probability
            #print(idx,word)
            try:#if we've seen the word before
                tag_list[idx] = list(word_dict[word].keys())
            except KeyError:#if we have not seen this word before
                if word[0].isupper():#-->If the first char of unknown word is capitalized, call it NP
                    tag_list[idx] = ['NP']
                    word_dict[word] = {'NP':1}
                elif word[-1] == 'y' or word[-2:] == 'al': #if it ends with y or al call it adjective
                    tag_list[idx] = ['JJ']
                    word_dict[word] = {'JJ':1}
                elif any(char.isdigit() for char in word):#if any chars are numeric
                    tag_list[idx] = ['CD']#we'll say it's a number
                    word_dict[word] = {'CD':1}
                else:#otherwise it's  a noun
                    tag_list[idx] = ['NN']
                    #We will say it is a noun
                    word_dict[word] = {'NN':1}
                    #We will add the word to our corpus as a noun to avoid future Key Errors 
            for tag in tag_list[idx]:
                #for every tag this word has
                maximize = []
                for prev_idx,prev_prob in enumerate(score[idx-1]):
                    #for every probability in the previous word
                    maximize.append(prev_prob * (pos_bi_gram_dict[(tag_list[idx-1][prev_idx],tag)] / pos_dict[tag_list[idx-1][prev_idx]]))
                    #append to list the prob of the previous word/tag pair * the current tag given previous tag
                prob_of_tag = (word_dict[word][tag]/pos_dict[tag])*max(maximize)
                #print(idx)
                this_word_prob_list.append(prob_of_tag)
                max_back_ptr.append(maximize.index(max(maximize)))
            #score.append()
            score.append(this_word_prob_list)
            back_ptr.append(max_back_ptr)
        #Sequence Identification
        best_tags_list_reverse = []
        reverse_index = len(sentence) - 1
        for word_r in reversed(score):
            highest_score_idx = word_r.index(max(word_r))
            best_tags_list_reverse.append(tag_list[reverse_index][highest_score_idx])
            reverse_index = reverse_index - 1
        #print(best_tags_list_reverse)
        sentence_tags = list(reversed(best_tags_list_reverse))
        predicted_tags.append(sentence_tags)
    return predicted_tags

#This function generates all missing pos bi-grams and inserts them into the training set with occurence 1. This prevents key errors from popping up with unseen part of speech bi grams.      
def create_all_pos_bi_gram_permuations(pos, pos_tup):
    pos_list = list(pos.keys())
    all_tags_tup = product(pos_list,pos_list)
    for tup in all_tags_tup:
        if tup not in pos_tup:
            #If we haven't seen a bi-gram before
            pos_tup[tup] = 1
            #add it to the dicionary with count of 1
    return pos_tup

#This functin takes two lists of lists. Actual contains the actual tags stripped from the test set. Predicted contains the predicted tags from viterbi/frequency count.
def calculate_accuracy(actual, predicted):
    correct = 0
    total = 0
    wrong = []
    for sentence_idx in range(len(actual)):
        for word_idx in range(len(actual[sentence_idx])):
            total = total + 1
            if(actual[sentence_idx][word_idx] == predicted[sentence_idx][word_idx]):
                correct = correct + 1
            else:#I used wrong[] to identify shortcomings in the program as listed in my writeup
                wrong.append((actual[sentence_idx][word_idx],predicted[sentence_idx][word_idx]))
    print(str(round((correct/total)*100, 3)),'%')
    return wrong

#This writes the testing data words and predicted pos to file
def write_to_file(test_words, predicted_tags):
    template = "{0:15}{1:5}"
    f = open("POS.test.out", "w")
    for sentence_idx in range(len(test_words)):
        for word_idx in range(len(test_words[sentence_idx])):
            line_tup = (test_words[sentence_idx][word_idx], predicted_tags[sentence_idx][word_idx])
            f.write(template.format(*line_tup)+'\n')

#This block of code grabs file names from command line 
able = str(sys.argv).split(',') 
regex = re.compile('[^a-zA-Z.]') 
training_file = regex.sub('', able[1]) 
testing_file = regex.sub('', able[2])  
            
corp = extract_tuples_train(training_file)#This preprocesses the training data
words, pos_tup, pos = create_dictionaries(corp)#This represents the training data as three dictionaries
pos_tup = create_all_pos_bi_gram_permuations(pos, pos_tup)#This fills in missing pos bi-grams with count 1
test_sentences, test_tags_actual = extract_tuples_test(testing_file)#this prepares the testing data
predicted_viterbi = viterbi_algorithm(words, pos_tup, pos, test_sentences)#This predicts the tags of the test data
wrong = calculate_accuracy(test_tags_actual, predicted_viterbi)#This calculates the accuracy of the tags
write_to_file(test_sentences, predicted_viterbi)#This writes the word+predicted pos to file
