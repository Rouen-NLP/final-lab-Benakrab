# -*- coding: utf-8 -*-
import codecs
import csv
import re
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import nltk


def lines_to_line(path):
    """ input : path of a text file
        output : extracted text after some preprocessing """
    file = codecs.open(path, 'r', "utf-8")
    lines = file.readlines()
    text = " ".join([line.strip() for line in lines])
    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    text = " ".join([word.lower() for word in text.split() if word.lower()
                     not in stopwords])
    # remove some special and control characters using regular expression
    regex = re.compile(r'[\n|\r|\t|$|£|&|(|*|%|@|)|\
                          "|“|,|:|.|;|+|?|!|~|{|}|§|#|"]')
    text = regex.sub(" ", text)
    # remove words with less than four characters
    text_recovered = " ".join([word for word in text.split()
                               if len(word) >= 4])
    return text_recovered


def extract_text(img_csv, txt_csv, repo):
    """ inputs : two csv files : source and target
        the first contains the image paths and their classes
        the seconde is the csv file produced that contains
        the text paths, the classes and the extracted text
        outputs : list of empty files, length of the list """
    csvinput = codecs.open(img_csv, 'r', "utf-8")
    csvoutput = codecs.open(txt_csv, 'w', "utf-8")
    writer = csv.writer(csvoutput, delimiter='\t')
    reader = csv.reader(csvinput, delimiter=',')
    # save the paths of the empty text files
    NAN = []
    # loop over the input csv file
    for i, row in enumerate(reader):
        # change the image path to a text path
        row[0] = row[0].replace("jpg", "txt")
        # create the headers
        if i == 0:
            writer.writerow(["txt_path", row[1], "text"])
        # extract the text
        else:
            text = lines_to_line(repo + row[0])
            # if missing values are detected
            if text == "":
                NAN.append(row[0])
            # create the csv file with the no empty extrcated values
            else:
                writer.writerow([row[0], row[1], text])
    csvinput.close()
    csvoutput.close()
    return NAN, len(NAN)


if __name__ == "__main__":
    missing_data, length = extract_text(sys.argv[1], sys.argv[2], sys.argv[3])
    print("Nomber of empty files : %d" % length)
    # create pandas dataframe
    df = pd.read_csv(sys.argv[2], sep="\t")
    # Split the dataset in a stratified way
    X_train, X_test, Y_train, Y_test = train_test_split(
            df["text"], df["label"], stratify=df["label"],
            test_size=0.2, random_state=0)
    # create vectors
    vectorizer = CountVectorizer(max_features=2600)
    vectorizer.fit(X_train)
    X_train_counts = vectorizer.transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    # TF-IDF representation
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_test_tf = tf_transformer.transform(X_test_counts)
    # train our SVM classifier with optimized parameters
    final_clf_svm_tf = LinearSVC(
            multi_class="crammer_singer",
            C=1, loss='squared_hinge',
            max_iter=1000, penalty='l1',
            tol=0.001).fit(X_train_tf, Y_train)
    print("Accuracy on the test dataset = %.2f"
          % final_clf_svm_tf.score(X_test_tf, Y_test))
    y_pred = final_clf_svm_tf.predict(X_test_tf)
    print("Classification Report")
    print(classification_report(Y_test, y_pred))
    print("Confusion Matrix")
    print(confusion_matrix(Y_test, y_pred))