# Context and Methodology

In the famous lawsuit against American tobacco companies, 14 million documents were collected and digitized. In order to facilitate the use of these documents by the lawyers, we are in charge of setting up an automatic classification task. <br>
We have in our possesion a random sample of the documents that contains : <br>
    * Images of documents in jpeg format <br>
    * The extracted text from images by an optical character recognition technology <br>
    * A csv file that contains the paths of images and their corresponding classes <br>
In this work, the methodology followed is : <br>
    * First, we will create a *Pandas* dataframe that contains, for each row, the path to the text file, the class of the document and the extracted text. As may be expected, some preprocessing operations will be performed <br>
    * After the phase of data acquisition, we will investigate and get more insights about our 
    dataset using charts and descriptive statistics in order to achieve a data split operation 
    that takes into account the distribution of our classes <br>
    * The modelisation stage consists of selecting a model that performs well according to 
    some metric, the accuracy for instance. Naturally, hyperparameters optimization and 
    error analysis will be realized too <br>
    * After choosing a model, we will create a python script that can be run through the command 
    line and performs the previous steps. <br>
    * Finally, some limits and perspectives will be presented.
    
# Data Extraction and Processing

Among the difficulties encountered during the extraction phase : <br>
    * The provided csv file contains the paths to the images and not to the text files <br>
    * Our text files contain many text lines <br>
    * The text files contain stopwords and special characters <br>
    * We detected missing data from a very few text files
To fix these issues, we defined a function named *extract_text* that : <br>
    * loops over the input csv file <br>
    * Changes the image paths to a text paths <br>
    * Converts the text, for each document, to a single line ; after removing stopwords,
    special and control characters <br>
    * Extracts the text <br>
    * Handles the empty files by ignoring them <br>
    * Saves the paths of the empty files to keep a state record <br>
    * Creates the ouput csv file that contains the paths, the classes and the extracted text <br>
    * Return the list of empty files and its length <br>
As a result, we detected 32 empty text files.
    
# Data Exploration

To get some insights about our extracted data, we converted the produced csv file to a pandas dataframe and realized the following : <br>
    * Verify the dimensions <br>
    * Verify that there is no missing values <br>
    * Plot the word frequency histogram that shows the most common words in our dataset <br>
    * Produce some charts that describe the distribution of our classes <br>
    * Take into account this distribution when spliting our data to train, dev and test <br>
    (stratify)
Our dataset contains 3450 observations with no missing values. The most common word in our documents is *Tobacco* ; returning clearly the context of our data. The distribution of our classes is unbalanced. A Jupyter notebook was produced showing explicitly the results.

# Building models
In this stage, we build a baseline model (e.g. Bayes Naive classifier) and performed our first classifcation attempt using Bag-of-words model and TF-IDF representation. After these experiments, we added more complexity by choosing other classifiers : a Linear SVM and a Multilayer Perceptron.<br>
The Linear SVM showed better accuracy on the TF-IDF representation. Therefore, we will switch to the phase of hyperparameters optimization in order to enhance the performance of this classifier. Performing a *Grid Search Cross-Validation* is a tedious task, especially computationally. So we will just manipulate a few hyperparameters of the SVM classifier. In fact, the value of *max_features* was chosen, in some how, according to the spirit of a cross validation procedure. So, we preserve this value and try to optimize the other parameters liked to the model. The experiments are detailed in the notebook attached to this work.<br>
A Neural Network classifier is build too. The results achieved are reasonably good comparing to the size of our dataset. Indeed, to do better, our neural network needs more data because it performs two tasks : features learning and classification.

# Limits and Perspectives

The main limits are due to the size and the distribution of our dataset that is unbalanced. Indeed, the data in possession is not large enough to train perfectly our neural network. As a solution, one
can handle these issues by : <br>
    * making more preprocessing. for instance, we can consider "smoking" and "smoke" as 
    identical words. <br>
    * collect more data, especially from the minority classes <br>
    * performing data augmentation in the case where collecting more data can not be done <br>
    * using downsampling and upsampling methods to deal with unbalanced classes, or assign weights
    to each class according to their frequencies <br>
    * training models that deal with such cases (e.g. random forest class_weight="balanced") <br>
    * ....
More work can be done on the data side. In the same way, we can enhance the present work from the models side. For instance :
    * Optimize the hyperparameters, using *Grid Search Cross-Validation*, attached to the
    vectorizing process (e.g. max_features, max_df, min_df, use_idf, ngram_range, ...) <br>
    * Do the same for the neural network and the MLP classifiers <br>
    * Train our SVM classifier with more data, as showed in the learning curves <br>
    * ..........
    
# Attached Files

There is three more files attached to this report : a Jupyter notebook that contains more expressed details about the experiments and the tasks performed, a Python script that takes arguments from the command line and perform a classification task using our SVM classifier ; and a text file showing the versions of the libraries.<br>
To execute the script from the command line, we can use the following : <br>
*python final_lab_Tobacco.py "Tobacco3482.csv" "Tobacco_text.csv" "data/"* <br>
The final argument is the folder where the text files are saved. 
