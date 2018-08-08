----------------------------------------------------------------Guess Numer--------------------------------------------------------------
#== == == == == == == == == == == == == == == == == == == == == NAMEPRINTAND
#CONCATINATE == == == == == == == == == == == == == == == == == =
'''for x in range(6):

    print("taha")

    fruits = ["apple", "banana", "cherry"]
    for x in fruits:
        print(x)

        thisdict = {
            "apple": "green",
            "banana": "yellow",
            "cherry": "red"
        }
        print(thisdict)
'''
#= == == == == == == RANDOM NUM GAME(WITHOUt LIMIT) == == == == == == == == =
'''
import random

n1 = random.randrange(1, 10)
while True:
    print('Guess a number between 1 and 10')
    guess = input()
    i = int(guess)
    if i == n1:
        print('You won!!!')
        break
    elif i < n1:
        print('Guess Larger')
    elif i > n1:
        print('Guess Smallest')
'''
#== == = == == == == == =WITH LIMIT == ==== == == == == =
import random

n1 = random.randrange(1, 10)

for value in range(5):
    print('Guess a number between 1 To 10')
    guess = input()
    i = int(guess)
    if i == n1:
        print('You won!!!')
        break
    elif i < n1:
        print('Guess Larger')

    elif i > n1:
        print('Guess Smaller')

    elif i != n1:

        print('You Lost')

else:
    print('You Lost')




------------------------------------------------------------------Steeming------------------------------------------------------------
import nltk
from nltk.stem import PorterStemmer


pc = PorterStemmer()

#exp_1 = ["python","pythoner","pythoning","pythonerd","pythonly"]
exp_1 = ["it is very important to be pythonly while you are pythoning with pyhton. or pythoners have pythoned poorly atleast ones"]
#words = word_tokenize(exp_1)


for w in exp_1:
    print(pc.stem(w))

------------------------------------------------------------------Tokenize------------------------------------------------------------
import nltk
from nltk import sent_tokenize,word_tokenize


from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer


#example_text = "Hellp!,How are you doing? i hope that you are well and doing grate at the same time.My name is asad and i am very "

#print(sent_tokenize(example_text))

train_text=state_union.raw("2005-GWBush.txt")
#print(train_text)

#nltk.download('state_union')


sample_text =state_union.raw("2006-GWBush.txt")
custom_txt_tok = PunktSentenceTokenizer(sample_text)
tokenized = custom_txt_tok.tokenize(sample_text)

print(custom_txt_tok)
def process_cxt():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagg = nltk.pos_tag(words)
            print(tagg)



    except Exception as e:
        print(str(e))


process_cxt()
#nltk.download('averaged_perceptron_tagger')


------------------------------------------------------------------6-d------------------------------------------------------------
#Write down a python code to classify the text of the movie reviews from the corpus data set and identify the word appearing Least of the time.

import nltk
import random

#random.download()
#''''
from nltk.corpus import movie_reviews  # movie review data set

doc1 = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]

random.shuffle(doc1)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(30) == False)
#'''
------------------------------------------------------------------6-c------------------------------------------------------------
#Write down a python code to classify the text of the movie reviews from the corpus data set and identify the word appearing most of the time.
import nltk
import random
from nltk.corpus import movie_reviews  # movie review data set

doc1 = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]

random.shuffle(doc1)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(30))

------------------------------------------------------------------6-b------------------------------------------------------------

#Write down a Python code to take a user define sentence and identify the named entity.
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = "Hello!, My name is Ali Akhbar Siddiqui. I am here to teach a course" \
              " of Artifical Intelligence in Iqra University."

custon_sent_tok = PunktSentenceTokenizer(train_text)
tokenized = custon_sent_tok.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize((i))
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()
            print(namedEnt)
    except Exception as s:
        print(str(s))

process_content()


------------------------------------------------------------------6-a------------------------------------------------------------
#Write down a Python code to determine the named entity in the text.
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custon_sent_tok = PunktSentenceTokenizer(train_text)
tokenized = custon_sent_tok.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize((i))
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            #namedEnt.draw()
            print(namedEnt)
    except Exception as s:
        print(str(s))
process_content()


------------------------------------------------------------------7-b------------------------------------------------------------
import nltk
import random
from nltk.corpus import movie_reviews

doc1 = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(doc1)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:300]


def find_features(doc1):
    words = set(doc1)
    features = {}  # empty dictonary

    for w in word_features:
        features[w] = (w in words)

    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))


------------------------------------------------------------------7-a------------------------------------------------------------
# In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature
# in a class is unrelated to the presence of any other feature. For example, a fruit may be
# considered to be an apple if it is red, round, and about 3 inches in diameter.

import nltk
import random
from nltk.corpus import movie_reviews

doc1 = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(doc1)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:300]


def find_features(doc1):
    words = set(doc1)
    features = {}  # empty dictonary

    for w in word_features:
        features[w] = (w in words)

    return features


print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in doc1]
train_set = featuresets[:190]
test_set = featuresets[190:]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print('Naive Bayes Algorithm Accuracy: ', (nltk.classify.accuracy(classifier, test_set)) * 100)
classifier.show_most_informative_features(15)


------------------------------------------------------------------8-d------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

male = 1
female = 0

features = np.array([[6, 180, 12],
                     [5.92, 190, 11],
                     [5.58, 170, 12],
                     [5.92, 165, 10],
                     [5, 100, 6],
                     [5.5, 150, 8],
                     [5.42, 130, 7],
                     [5.75, 150, 9]])

labels = np.array([male, male, male, male, female, female, female, female])
person = np.array([[5.73, 170, 8]])

classifier = GaussianNB()
classifier.fit(features, labels)

prediction = classifier.predict(person)

if prediction == 1:
    print('New Entry is Male')
elif prediction == 0:
    print('New Entry is Female')



------------------------------------------------------------------8-c------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


features = pd.DataFrame()
labels = pd.DataFrame()
person = pd.DataFrame()

# Generate or acquire list of all features
height = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
weight = [180,190,170,165,100,150,130,150]
foot_size = [12,11,12,10,6,8,7,9]

# Labels
male = 1
female = 0
label = [male,male,male,male,female,female,female,female]


while(True):

    labels = np.array(label)
    # Writing features into Arrays
    features = pd.DataFrame()
    features['Height'] = height
    features['Weight'] = weight
    features['Foot Size'] = foot_size
    features_1 = np.array(features)
    print(features_1, '\n')

    # Define Classifier
    classifier = GaussianNB()

    # Train Classifier using name.fit()
   # classifier = classifier.fit(features_1, labels)

    classifier1 = tree.DecisionTreeClassifier()
    classifier = classifier1.fit(features_1, labels)

    # Take input for the next entry from user
    new_h = input('Enter Height of the New Entry: ')
    new_w = input('Enter Weight of the New Entry: ')
    new_fs = input('Foot Size of the New Entry: ')

    # Writing new entry into Arrays
    person['Height'] = [new_h]
    person['Wight'] = [new_w]
    person['Foot_Size'] = [new_fs]
    person_1 = np.array(person)

    # Predict the class or label using GaussianNB
    prediction = classifier.predict(person_1)

    if prediction == 1:
        label.append(male)
        print('\n','New entry is a Male')
    elif prediction == 0:
        label.append(female)
        print('\n','New entry is a Female')

    height.append(new_h)
    weight.append(new_w)
    foot_size.append(new_fs)

------------------------------------------------------------------8-b------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import neighbors


# Create an empty dataframe
features = pd.DataFrame()
labels = pd.DataFrame()
person = pd.DataFrame()

male = 1
female = 0

# Create our target variable
labels = np.array([male,male,male,male,female,female,female,female])

person['Height'] =    [6]
person['Weight'] =    [170]
person['Foot_Size'] = [8]
person = np.array(person)

# Create our feature variables
features['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
features['Weight'] = [180,190,170,165,100,150,130,150]
features['Foot_Size'] = [12,11,12,10,6,8,7,9]
print(features,"\n",'\n')
features_1 = np.array(features)
#print(features,"\n",'\n')

#classifier = GaussianNB()
#classifier2 = neighbors.KNeighborsClassifier()
#classifier = classifier2.fit(features_1, labels)

classifier1 = tree.DecisionTreeClassifier()
classifier = classifier1.fit(features_1, labels)


prediction = classifier.predict(person)

if prediction == 1:
    print('New entry is a Male')
elif prediction == 0:
    print('New entry is a Female')

------------------------------------------------------------------8-a------------------------------------------------------------
import pandas as pd
import numpy as np

# Create an empty dataframe
data = pd.DataFrame()

# Create our target variable
data['Gender'] = ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female']

# Create our feature variables
data['Height'] = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
data['Weight'] = [180, 190, 170, 165, 100, 150, 130, 150]
data['Foot_Size'] = [12, 11, 12, 10, 6, 8, 7, 9]

# View the data
# print(data)

# Create an empty dataframe
person = pd.DataFrame()

# Create some feature values for this single row (consider this to be a new entry in tje given data set)
person['Height'] = [5.73]
person['Weight'] = [170]
person['Foot_Size'] = [8]

# View the data
# print(person)

# Number of males
n_male = data['Gender'][data['Gender'] == 'male'].count()

# Number of males
n_female = data['Gender'][data['Gender'] == 'female'].count()

# Total rows
total_ppl = data['Gender'].count()

# Number of males divided by the total rows (Total Entries)
P_male = n_male / total_ppl

# Number of females divided by the total rows
P_female = n_female / total_ppl

# Group the data by gender and calculate the means of each feature
data_means = data.groupby('Gender').mean()

# View the values
# print(data_means)

# Group the data by gender and calculate the variance of each feature
data_variance = data.groupby('Gender').var()

# View the values
print(data_variance)

# Means for male
male_height_mean = data_means['Height'][data_variance.index == 'male'].values[0]
male_weight_mean = data_means['Weight'][data_variance.index == 'male'].values[0]
male_footsize_mean = data_means['Foot_Size'][data_variance.index == 'male'].values[0]

# Variance for male
male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0]
male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]
male_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'male'].values[0]

# Means for female
female_height_mean = data_means['Height'][data_variance.index == 'female'].values[0]
female_weight_mean = data_means['Weight'][data_variance.index == 'female'].values[0]
female_footsize_mean = data_means['Foot_Size'][data_variance.index == 'female'].values[0]

# Variance for female
female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0]
female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]
female_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'female'].values[0]


# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):
    # Input the arguments into a probability density function
    p = 1 / (np.sqrt(2 * np.pi * variance_y)) * np.exp((-(x - mean_y) ** 2) / (2 * variance_y))

    # return p
    return p


# Numerator of the posterior if the unclassified observation is a male
a1 = P_male * p_x_given_y(person['Height'][0], male_height_mean, male_height_variance) * \
     p_x_given_y(person['Weight'][0], male_weight_mean, male_weight_variance) * \
     p_x_given_y(person['Foot_Size'][0], male_footsize_mean, male_footsize_variance)

# Numerator of the posterior if the unclassified observation is a female
a2 = P_female * p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * \
     p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * \
     p_x_given_y(person['Foot_Size'][0], female_footsize_mean, female_footsize_variance)

if a1 > a2:
    print('New Entry is a Male')
elif a1 < a2:
    print('New Entry is a Female')








------------------------------------------------------------------clus-1------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

from sklearn.cluster import KMeans

x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x,y)
plt.show()

------------------------------------------------------------------clus-2------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')


x = [1, 5, 1.5, 8, 1, 9]
y = [2, 8, 1.8, 8, 0.6, 11]

plt.scatter(x, y)
plt.show()

X = ([[1, 2],
      [5, 8],
      [1.5, 1.8],
      [8, 8],
      [1, 1.6],
      [9, 11]])

kmean = KMeans(n_clusters=2)
kmean.fit(X)

centroids = kmean.cluster_centers_
labels = kmean.labels_

print(centroids)
print(labels)

colors = ['g.', 'r.', 'y.']

for i in range(len(X)):
    print('Co-ordinates: ', X[i], 'label: ', labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()

    
