# AI Junior Developer (Intern) Test 
Devloped a basic NLP chatbot for answering the basic science question  

## Requirement
The current project has the blueprint structure of an AI App. 

Your mission is to implement an 💬NLP chatbot **answering questions about science**. 

You will add your logic to the `main.py` file inside the `execute` function. 

## Library used

* The numpy library and alias it as np for convenience
* The Natural Language Toolkit library
* The string module, which provides various string manipulation functions
* The TfidfVectorizer class from the scikit-learn library
* The cosine_similarity function from the scikit-learn library


## The entire process 
 Get raw data -> convert to lower case -> Sentence tokenization -> word tokenization -> TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer->check the cosine similarity between user input and the list of sentence -> get second highest values of the similarity between input and the sentence -> get the sentence index value which highest based on previous value of similarity -> get the output  





## Input
<img width="546" alt="Screenshot 2023-03-04 at 15 08 39" src="https://user-images.githubusercontent.com/91294460/222913713-a0b07829-9228-4cc6-86a3-c36517a9a374.png">

## Output
<img width="777" alt="Screenshot 2023-03-04 at 15 08 57" src="https://user-images.githubusercontent.com/91294460/222913726-ca5fe104-30e7-42fa-aead-6a42272aa460.png">


