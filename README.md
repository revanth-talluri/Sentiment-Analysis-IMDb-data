# Sentimental Analysis on Imdb reviews using Deep Learning
Natural Language Processing (NLP) is one of the fastest growing research areas in the Machine Learning field. Interest in NLP began in the early 1950's when Alan Turing published his paper '[Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238)' and from which the famous [Turing-test](https://en.wikipedia.org/wiki/Turing_test) emerged. With the ongoing growth in online activity, huge amount of data is generated everyday. And since most of this data is unstructured, it is quite difficult to draw meaningful insights from it. And this is where NLP comes in. The NLP techniques help machine understand the information and increases the efficiency.

Some of the famous applications of NLP are Speech Recognition, Chatbots, Text Classification, Sentimental Analysis etc. Let's see an application of Sentimental Analysis here.

Sentimental Analysis is the interpretation and classification of emotions within text data. Sentiment analysis (also known as opinion mining or emotion AI) refers to the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information (Source: Wikipedia).

The dataset taken here is from the [Stanford Imdb dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The dataset contains a total of 50,000 reviews from Imdb with 50/50 split between train and test sets. And within each of these, we have 50/50 split for positive and negative reviews. Let's have a look at how we can use this data and predict whether if a review is positive or negative. We will be using Deep Learning techniques (Artificial Nueral Networks and Convolution Nueral Networks) to predict the sentiment. There are 5 main steps involved here.

- Reading the .txt files and putting them in a dataframe
- Preparing the corpus for train and test set
- Tokenize the dataset and finding the vocabulary length
- Word Embedding and padding
- Building the model

## Reading the .txt files and putting them in a dataframe
Since there are a total of 50,000 reviews, all of them are not in the same folder. They are seperated into train and test set folders and also the postive and negative reviews are in seperate folders in each train and test set folders. And each review is in a seperate .txt file. We'd have to read every .txt file here. 

Since the location of these folders are in a different path than the .py script file we are runnnig, we will use the 'os' library. And this part of code is not generalized and check of the path given in the code matches the path of the files. We have a total of 4 folders to read and we have 4 different paths. And after reading all the files, make sure that the final path matches the script file path. We will create a train_file and test_file list and store all the reviews in the respective lists. 

Now let's put this into dataframe. The 'review' will be our feature and the 'sentiment' will be our target. Since we don't have 'sentiment' list, let's create one using the np.ones/np.zeros. We have now both our feature and target columns and we will use these to form train_df and test_df. In these dataframes, the first 12,500 reviews are positive and the next 12,500 reviews are negative. This might create a bias while training the model, so let's shuffule all the rows in these dataframes.

## Preparing the corpus for train and test set
After gettig our dataframes, we will now move on to create a corpus. Let's see train_df for now. We have a dataframe with 25,000 rows and 2 columns. The 'Review' column contains all the reviews and the 'Rating' column says whether the review is positive or negative. Our plan here is to identify the words that are associated with each sentiment and then use that information to predict a new review. Let's take a review from the 'Review' column and check it.

This review might have small and capital letters, punctuations, exclamations and the most common occuring words in the english language (stopwords). We will try to adjust all reviews to fit the same pattern. First, the stopwords, the most occuring common words are in both the positive and negative sentiment. We will remove such words from the review. Next we will convert all the capital letters to small and remove any and all punctuations, exclamations and other symbols, numbers from the review. We will apply stemming to reduce the words to their root word. For example 'love', 'loving', 'loved' have the same root word 'love' and convey the same meaning. Keeping all such words in the corpus will increase the memory usuage and also take up more time during our classification. So we can reduce every word to their root word. And also we can check if the word has the correct spelling or not using the 'spellcheker' library. There are several spell checking algorithms/libraries, the one used here is based on [Peter Norvig’s blog post](https://pypi.org/project/pyspellchecker/) on setting up a simple spell checking algorithm. 

But since the 'stemming' and 'spell check' has to run on every word in the entire dataset, it takes up a huge amount of time to run. So that part of the code will be commented out here, but you can uncomment it and check the differene in corpus if interested.

After applying all the above steps, we now have a corpus in a simpler format.

## Tokenize the dataset and finding the vocabulary length
We will move onto the tokenization. The 'tokenization' process splits the sentence into slices of tokens. Let's create an empty list to store all the words in the dataset. We will go through each review and apply the 'word_tokenize' function and then store all the words in the list. We then use a 'set' on this list to find the total number of unique words in this entire dataset. This will be our vocabulary length. 

In this case, we will round-off our vocabulary length to the next nearest hundred.

## Word Embedding and padding
So far, we still have text as our feature, but a machine can't understand the text. So let's convert the text into real valued numbers using the 'one_hot' funtion from keras.text.preprocessing. What this does is that it assigns a real value to each word and all the same words have the same value, like a cipher. 

Now our text in converted into numbers, but we still have a problem. Our nueral network layer expects same input dimension every time, but here we have reviews with varying lengths. To overcome this, we will pad the reviews with '0's. Now we can pad either before or after the review. Here we will be using 'post' padding. Now how do we decide on the length of this review that we are giving to nueral network?
We will plot a box plot of our embedded sentences. This plot can show us the outliers and we can see the data based on quartiles. #looking a box and whisker plot for the review lengths in words, we can probably see an exponential distribution that we can probably cover the mass of the distribution with a clipped length of 2500 words. So our max_words is 2500.

## Building the model
Now before we move any forward, there is one final thing to do, 'Word Embeddings'. This is a technique where words are encoded as real-valued vectors in a high-dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space. Discrete words are mapped to vectors of continuous numbers. This is useful when working with natural language problems with neural networks and deep learning models are we require numbers as input.

Keras provides a convenient way to convert positive integer representations of words into a word embedding by an Embedding layer.

The layer takes arguments that define the mapping including the maximum number of expected words also called the vocabulary size (e.g. the largest integer value that will be seen as an integer). The layer also allows you to specify the dimensionality for each word vector, called the output dimension. Source: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

We will be using a 32 dimension in our model. The output of this layer is a matrix of size 2500 x 32 for each review. But our nueral netowrk expects a 1D input dimension. So we will faltten our matrix and then pass our input. In our model we will be using a 2 hidden layer(other than embedding layer) network with 64 and 32 units respectively and the output consists of 1 node, which will tell us if the sentiment is either '1' (positive) or '0' (negative). 

Now usually we will be doing this using an Artificial Nueral Network. But we can also implement this using an Convolution Nueral Network. Convolutional neural networks were designed to honor the spatial structure in image data whilst being robust to the position and orientation of learned objects in the scene. This same principle can be used on sequences, such as the one-dimensional sequence of words. The same properties that make the CNN model attractive for learning to recognize objects in images can help to learn structure in paragraphs of words, namely the techniques invariance to the specific position of features.

In this case, we will build both the models and see the results.

## Results
- The ANN model resulted in an accuracy of 86.62%
- The CNN model resulted in an accuracy of 85.79%

## Note
Both our models has an accuracy of over 85% and this is a good start. We can improve the accuracy by trying to optimze the hyperparameters of the model, no. of hidden layers, batch size, epochs, nodes in the hidden layers, optimizer and loss function.

We can also use Classification techniques to do the Sentimental Analysis. You can check this from my other repo. Note that the dataset used for this repo is a very small one, containing only 2000 reviews for train and test sets and the accuracy there is between 70% to 75% for several models. We can improve the accuracy if more data is available.


