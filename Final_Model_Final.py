#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 12:26:29 2018

@author: GabrielleChester
"""
#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
import os
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.exceptions import DataConversionWarning


#This function scans through all the files in the DataSet folder
def fileFinder(file):
    global path
    folder = 'DataSet/'
    path = folder + file
    return path

#This function finds the median of the coefficient of friction column and stores it in the list med=[], for each file in the path
def stat(path):
    global med
    med=[]
    names = ['Fast data in','Time','This Step','Step Time','Test Time','Friction Force (N)','Frequency (Hz)','Load (N)','Specimen Temp (∞C)','FrictionCoeff','Friction Range (N)','Contact Pot (mV)','Force Input (Volts)']
    data=pd.read_csv(path, names=names, engine='python')
    df=pd.DataFrame(data)
    data1=df.iloc[:,9]
    df1=pd.DataFrame(data1)
    df1.dropna(how='any', inplace=True)
    df2=df1[df1.FrictionCoeff.str.contains("Friction Coeff") == False]
    x=df2.median()
    med.append(x.values)
    return med

#This function drops any row which has missing data from it and any columns specified in main and creates the classes for the
# coefficient of friction to be set
def DropRows(df):      
    df.drop(features_to_delete, axis=1, inplace=True)
    print(df.shape)              
    df.dropna(how='any', inplace=True)
    print("New shape with empty rows removed")
    
    conditions = [
            (df['Median CoF'] >= 0.3),
            (df['Median CoF']<= 0.1)]
    choices = [1, -1]
    df['Classify'] = np.select(conditions, choices, default=0)
    print(df.shape)
    return df;

#This function splits the data set randomly into 70% for training and 30% for training
def splitData(df,X,y):              
	split_test_size = 0.30
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = split_test_size, random_state = 42)	
	return X_train,X_test,y_train,y_test   

#This function builds a library of materials for the model to understand
def MaterialProcessor(g):
    global df
    df1=g.replace('Colsibro ',1)
    df2=df1.replace('Sorona PTT',2)
    df3=df2.replace('UHMWPE',3)
    df4=df3.replace('Duratron P(A-I)',4)
    df5=df4.replace('Nylatron NSM',5)
    df6=df5.replace('PEEK',6)
    df7=df6.replace('PTFE',7)
    df8=df7.replace('Delrin',8)
    df9=df8.replace('PTT',9)
    df10=df9.replace('ZYTEL',10)
    df11=df10.replace('Al+Graphite',11)
    df12=df11.replace('Aluminium',12)
    df13=df12.replace('Delrin500AF',13)
    df14=df13.replace('Titanium',9)
    df15=df14.replace('DelrinNC010',10)
    df16=df15.replace('Dryslide',11)
    df17=df16.replace('PPS',12)
    df=df17.replace('Delrin500AF',13)
    print('Materials have been processed')
    return (df); 
   
#This function builds a library of lubricants for the model to understand
def LubricationProcessor(f):
    global df
    df1=f.replace('ISO10',1)
    df2=df1.replace('HEF submerged',2)
    df3=df2.replace('Dry',3)
    df4=df3.replace('absorption test',4)
    df5=df4.replace('HEF Drip',5)
    df6=df5.replace('ISO10 w 10% contam',6)
    df7=df6.replace('ISO 10 w 10% contam emulsified',7)
    df8=df7.replace('Dry Sliding',8)
    df9=df8.replace('PTT',9)
    df10=df9.replace('OilHef',10)
    df11=df10.replace('Distilled water',11)
    df12=df11.replace('POA',12)
    df13=df12.replace('PDMS',13)
    df14=df13.replace('Silicone',14)
    df15=df14.replace('HEF65',15)
    df=df15.replace('HEF',16)
    print('Lubricants have been processed')
    return (df);

#This is the start of the main script
if __name__ == "__main__":
    path = 'feature_set.csv' #This if the file where the features for tests are stored
    names = ['Sample No.','Lower Material','Upper Material','Temperature','Density','Hardness','Contact Pressure','Lubrication','Surface Roughness','Titanium Sample'] #set column names
    data = pd.read_csv(path, names=names) #Imports data files
    df = pd.DataFrame(data) #formats file into dataframe
    features_to_delete =['Sample No.', 'Density','Hardness','Titanium Sample'] #deletes columns not relevant to data set
    print("Original Shape") 
    print(df.shape) #print the orginal size of data set before cleaning
    
    median=[]  #empty list for median files to be stored in
    paths='DataSet/' #sets the path for where the files are stored
    files = os.listdir(paths) #defines directory
    files.remove('.DS_Store') #removes file
    files.sort() #sorts files into alphabetical order
    
    #this for loop uses the functions fileFinder and stat to go through all the files and store hte median coefficient of friction
    #in the dataframe.
    for f in files:
        fileFinder(f)
        stat(path)
        median.append(med)
    #creates new dataframe with the new column containing median CoF
    name = ['Median CoF']
    medDf=pd.DataFrame(median, columns=name)
    #combines the two datadrames
    result = pd.concat([df, medDf], axis=1)
    #writes result into new file
    directory='featureset.csv'
    result.to_csv(directory)
    
    df = DropRows(result) #runs the drop rows function

    MaterialProcessor(df) #processes the materials
    LubricationProcessor(df) #processes the lubricants 
    print(df) #prints dataframe to check if previous steps have been done correctly
   
    feature_col_names = ["Lower Material","Upper Material","Temperature","Contact Pressure","Lubrication","Surface Roughness"] #define which columns are features
    predicted_class_name = ["Classify"] #defines which column is being predicted

    X = df[feature_col_names].values #sets up features as x values
    y = df[predicted_class_name].values #sets up predicted y column
	
    X_train, X_test, Y_train, Y_test = splitData(df,X,y) #splits data
    print(df.shape)
    print("{0:0.2f}% in training set".format((len(X_train)/len(df.index))*100))  #prints the percentage split
    print("{0:0.2f}% in test set".format((len(X_test)/len(df.index))*100))
    print("")

    XTrain = pd.DataFrame(X_train, columns = feature_col_names)     #defines the training and test data into x and y
    XTest = pd.DataFrame(X_test, columns = feature_col_names)
    YTrain = pd.DataFrame(Y_train, columns = predicted_class_name)
    YTest = pd.DataFrame(Y_test, columns = predicted_class_name)
    
    #feature selection process
    sel=SelectKBest(k=4) #uses the selectKbest statistical test
    selected=sel.fit_transform(XTrain,YTrain) #fit and transform the data so that it applies the feature selection process
    print(selected.shape) #print shape of new dataframe
    test_selected=sel.transform(XTest)  #tranfrom test set so that it c=nly contains the same features
    
    clf = RandomForestClassifier(max_depth=None, min_samples_split=2,random_state=0) #define algorithm
    clf.fit(selected,YTrain.values.ravel()) #fit dataset to algorithm
    y_pred=clf.predict(test_selected) #predict samples in test set
  
    #print performance of model
    print("Random Forest Algorithm")
    print(confusion_matrix(YTest,y_pred))
    print(classification_report(YTest,y_pred))
    print("Accuracy: ")
    print(accuracy_score(YTest, y_pred))
    print("")
    

    print(clf.feature_importances_)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
   
    #This piece of code can be uncommented to produce a feature importances graph for the model 
    """
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(selected.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    import matplotlib.pyplot as plt
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
    """
    
    #This piece of code corresponds with a excel file which can be used by the user to enter feature and output a prediction, 
    #provided that the materials and lubricants used are ones within the library.
    """
    feature_col_names = ["Lower Material","Contact Pressure","Lubrication","Surface Roughness"]
    path2='Model_Prediction.csv'
    print("done")
    predict_data = pd.read_csv(path2, names=feature_col_names)
    df=pd.DataFrame(predict_data)
    
    MaterialProcessor(df)
    LubricationProcessor(df)
    print(df)
    
    y_pred_new=clf.predict(df)
    print(y_pred_new)
    """
