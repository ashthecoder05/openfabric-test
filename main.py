from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import numpy as np   # import the numpy library and alias it as np for convenience
import nltk          # import the Natural Language Toolkit library
import string        # import the string module, which provides various string manipulation functions
from sklearn.feature_extraction.text  import TfidfVectorizer # import the TfidfVectorizer class from the scikit-learn library
from sklearn.metrics.pairwise import cosine_similarity       # import the cosine_similarity function from the scikit-learn library




############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    nltk.download('punkt')# download the Punkt tokenizer, which is used for tokenizing sentences
    nltk.download('wordnet')# download the WordNet lexical database, which is used for synonym and antonym lookup
    nltk.download('omw-1.4')#download the Open Multilingual WordNet, a version of WordNet that includes multiple languages
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        f = open('./Sciencedata/data.txt','r',errors ='ignore') # Open the file located at './Sciencedata/data.txt' in read mode
        raw_doc = f.read() # Read the contents of the file into a string
        raw_doc = raw_doc.lower() # Converting entire text to lowercase
        sentence_tokens = nltk.sent_tokenize(raw_doc) # Tokenize the text into a list of sentences and a list of words
        word_tokens = nltk.word_tokenize(raw_doc)
        lemmer = nltk.stem.WordNetLemmatizer()
        
        # Define a function to lemmatize a list of tokens
        def LemTokens(tokens):
         return[
            lemmer.lemmatize(token) for token in tokens]
        
        # Define a dictionary of punctuation to remove from the text
        remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

        # Define a function to lemmatize and normalize a string of text
        def LemNormalize(text):
            return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))
        
        # Define a function to generate a response to a user input using cosine similarity
        def responseML(user_response):
            robol_response = ''

            # Initialize a TF-IDF vectorizer with the LemNormalize tokenizer and stop words set to 'english'
            TfidfVec = TfidfVectorizer(tokenizer = LemNormalize , stop_words = 'english')

            # Fit the vectorizer to the list of sentence tokens and transform the user input
            tfidf = TfidfVec.fit_transform(sentence_tokens)
            
            # Calculate the cosine similarity between the user input and each sentence in the list
            vals = cosine_similarity(tfidf[-1], tfidf)

            # Get the index of the sentence with the highest similarity to the user input
            idx = vals.argsort()[0][-2]

            # Flatten the similarity scores and sort them in descending order
            flat = vals.flatten()
            flat.sort()

            # Get the second highest similarity score
            req_tfidf = flat[-3]

            # Generate a response based on the similarity score
            if (req_tfidf == 0):
                robol_response = robol_response + "I am sorry. Unable to understand you!"
                return robol_response
            else:
                robol_response = robol_response+ sentence_tokens[idx]
                return robol_response 
     
        sentence_tokens .append(text)
        word_tokens = word_tokens + nltk.word_tokenize(text)
        
        # Generate a response to the user input using the responseML function
        response = responseML(text)
        output.append(response)

    return SimpleText(dict(text=output))
