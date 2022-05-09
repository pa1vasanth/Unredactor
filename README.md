# cs5293sp22-project3

### Project3: The Unredactor 
### Author: PAVAN VASANTH KOMMMINENI  

## Project Summary:
 In this project I'm building a model that discovers the name in the redacted data. The data is accessed from the link("https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv")

### Note: Use 16 Gb instance to execute the unredactor.py

Python Functions:  
1.Dataextract function  
2.Cleaning function  
3.vectorizer function  
4.train_features function  
5.filtering function  
6.Main function  

The cd is cs5293sp22-Project3.  

command line: pipenv run python unredactor.py 

Required Python Packages:  
1.Pandas: for the dataframes  
2.numpy: for the arrays  
3.spacy: for the labeled data creation.  
4.sklearn:  
	CountVectorizer: for the vectorization    
	RandomForestClassifier: For the model creation  
	scores: to find all the scores(f1,recall,precision)   
5.nltk:  
	sent_tokenize,word_tokenize: for the tokenization  
	stopwords: to remove stopwords  
	SentimentIntensityAnalyzer: for the sentiment analysis  

### Functions:  

Data extraction:(DataExtract)  
1.This functions loads the data from the url.  
2.In this function,I'm  dropping the na value rows  
3.It returns the data in form of dataframe object.  

Cleaning data:(Cleaning)  
1.This function removes stopwords and special characters from the data.  
2.It returns the cleaned data (data frame)  

Vectorization: (vectorizer)   
1.This functions is using CountVectorizer() for the vectorization.  
2.The vectorization is assigned to a dataframe.  
3.The merge of data and vectors will be returned.  

Feature extraction: (train_features)  
1.This functions extracts different features.  
2.Features:['Positivity score',redacted namelen,sentences,words and char-characters]  
3.The feaatures are initially assigned to dictionary and are merged to vectorized data.  

Filtering data: (Filtering)  
1.This functions filters the (train and validation) and (testing) data  
2.It will return both train and test data frames.  

Main function:  
1.This function call all the above the function.  
2.Randomforest model and prediction are included in this function.  
3.All the scores were printed in this function(0 to 1)  0.07:- 7%.  

### Test Cases:  

command line: pipenv run python -m pytest

test_DataExtract:  
This method is used to test data extraction functionality.It asserts true when the returned data is not equal to none.

test_cleaning:  
This method is used to test cleaning functionality.It asserts true when the returned data is not equal to none.

test_vectorizer:  
This method is used to test vectorizer functionality.It asserts true when the returned data is not equal to none.

test_train_features:  
This method is used to test test_train_features functionality.It asserts true when the returned data is not equal to none.

test_filtering:  
This method is used to test test filtering functionality.It asserts true when the returned data is not equal to none.

Assumptions:  
In the Random Forest Classifier I'm using default values except random state. The random_state is assigned to 25345.

Bugs:  
The model is not accurate while predicting the names in the redacted text.  
There are few rows contains 5-fields instead of 4-fields(avoided these lines by error_bad_lines=False)  

### Note: Use 16 Gb instance to execute the unredactor.py

References:  
https://www.baeldung.com/cs/multi-class-f1-score  
https://realpython.com/python-nltk-sentiment-analysis/  
https://www.datacamp.com/tutorial/random-forests-classifier-python  
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html  



	 
	

