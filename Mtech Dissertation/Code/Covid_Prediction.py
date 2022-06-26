from ast import expr_context
import random as r
import warnings
import pandas as pd
import os
import sys
#from goto import goto, label
#from sklearn.datasets import make_hastie_10_2

import DataCleaning
import Cleanup
import MachineLearning
import DisplayResult
import DisplayGraphs

warnings.filterwarnings("ignore")

ModelName, Train_Test_Split, Accuracy, FOneScore, Precision, Recall, Age = [],[],[],[],[],[],[]
ModelName_P, Train_Test_Split_P, Accuracy_P, FOneScore_P, Precision_P, Recall_P, Age_P = [], [],[],[],[],[],[]

#df = pd.read_csv('finalDataset')
tst_dt_size = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

Graphs = ['Point Graph', 'Bar Graph']
#df = pd.read_csv('final1.csv')
#classifier = KNeighborsClassifier()

if __name__ == "__main__":
    print('\n\n\n')
    print('$'*5,'='*20,'#'*5,'='*20,'$'*5)
    print('$'*5,' Dataset Operations ','$'*5)
    print('NOTE: Datasets are present in Dataset/ folder. Datasets are \n 1) PatientInfo.csv\n 2) Weather.csv \n 3) Region.csv \
    \nDataset operations include Merging Datasets on specific attributes and Cleaning dataset')
    DataCleaning.Merge_Datasets()
    print('#Dataset merging has been done. Dataset is ready to clean!\n\n')
    DataCleaning.Clean_Dataset()
    print('#Dataset has been cleaned. Dataset is ready to preprocess for training!')
    
    print('\n\n\n')
    
    print('$'*5,'='*20,'#'*5,'='*20,'$'*5)
    print('$'*5,' Preprocessing Dataset ','$'*5)
    
    filelist = [f for f in os.listdir('../Datasets/') if f.startswith("finalData") and not f.endswith("t.csv")]
    df_list = []
    for file in filelist:
        df1 = pd.read_csv('../Datasets/'+str(file))
        #print('../Datasets/'+str(file))
        df, label = MachineLearning.Preprocessing(df1)
        df_list.append(df)
        

                
    print('#Dataset has been preprocessed. Ready to Train Model!')

    print('\n\n\n')
    
    print('$'*5,'='*20,'#'*5,'='*20,'$'*5)
    print('$'*5,' Training Dataset ','$'*5)
    
    parameter_list = [{"model_name": "Random Forest Classifier", "model_file": "rfc", "train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5]},
                      {'model_name': 'K-Nearest Neighbour Classifier', 'model_file': 'knn', "train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5] },
                      {'model_name': 'Support Vector Machine', 'model_file': 'svm', "train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5]},
                      {'model_name': 'Naive Bayes Classifier', 'model_file': 'nb', "train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5]}, 
                      {'model_name': 'Neural Networks', 'model_file': 'nn', "train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5]},
                      {'model_name': 'Logistic Regression', 'model_file': 'lr', "train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5]}]
    
    print('#Training Datasets to get high performing Age')
    idx = [' ','1','2','3','4']
    
    for i, df in enumerate(df_list):
        for plist in parameter_list:
            plist["model_file"] = plist["model_file"]+str(idx[i])
            MachineLearning.Train_Model(df, label, plist)

    #Change
    filelist = [f for f in os.listdir('../Datasets/') if f.startswith("finalData") and f.endswith("t.csv")]
    df_list1 = []
    for file in filelist:
        df1 = pd.read_csv('../Datasets/'+str(file))
        # print('../Datasets/'+str(file))
        df, label = MachineLearning.Preprocessing(df1)
        df_list1.append(df)


    try:
        rfc_a = [MachineLearning.Accuracy[0],MachineLearning.Accuracy[5],MachineLearning.Accuracy[11],MachineLearning.Accuracy[17],MachineLearning.Accuracy[23]]                
    except:
        MachineLearning.recoverBackup()
    
    rfc_a = [MachineLearning.Accuracy[0],MachineLearning.Accuracy[6],MachineLearning.Accuracy[12],MachineLearning.Accuracy[18],MachineLearning.Accuracy[24]]
    knn_a = [MachineLearning.Accuracy[1],MachineLearning.Accuracy[7],MachineLearning.Accuracy[13],MachineLearning.Accuracy[19],MachineLearning.Accuracy[25]]
    svm_a = [MachineLearning.Accuracy[2],MachineLearning.Accuracy[8],MachineLearning.Accuracy[14],MachineLearning.Accuracy[20],MachineLearning.Accuracy[26]]
    nb_a = [MachineLearning.Accuracy[3],MachineLearning.Accuracy[9],MachineLearning.Accuracy[15],MachineLearning.Accuracy[21],MachineLearning.Accuracy[27]]
    nn_a = [MachineLearning.Accuracy[4],MachineLearning.Accuracy[10],MachineLearning.Accuracy[16],MachineLearning.Accuracy[22],MachineLearning.Accuracy[28]]
    lr_a = [MachineLearning.Accuracy[5],MachineLearning.Accuracy[11],MachineLearning.Accuracy[17],MachineLearning.Accuracy[23],MachineLearning.Accuracy[29]]

    # print("Best RFC: ",rfc_a.index(max(rfc_a)))
    # print("Best KNN: ",knn_a.index(max(knn_a)))
    # print("Best SVM: ",svm_a.index(max(svm_a)))
    # print("Best NB: ",nb_a.index(max(nb_a)))
    # print("Best NN: ",nn_a.index(max(nn_a)))
    # print("Best LR: ",lr_a.index(max(lr_a)))

    Idx_rfc = rfc_a.index(max(rfc_a))
    Idx_knn = knn_a.index(max(knn_a))
    Idx_svm = svm_a.index(max(svm_a))
    Idx_nb = nb_a.index(max(nb_a))
    Idx_nn = nn_a.index(max(nn_a))
    Idx_lr = lr_a.index(max(lr_a))                

    Idx_list = [Idx_rfc, Idx_knn, Idx_svm, Idx_nb, Idx_nn, Idx_lr]
    first,last = [0,0,0,0,0,0],[0,0,0,0,0,0]
    df_m3 = [df_list[0],df_list[0],df_list[0],df_list[0],df_list[0],df_list[0]]
    df_patient = [df_list1[0],df_list1[0],df_list1[0],df_list1[0],df_list1[0],df_list1[0]]
    age = [-1,-1,-1,-1,-1,-1]

    for i in range(len(Idx_list)):
        if Idx_list[i] == 4 :
            age[i] = 70
            df_m3[i] = df_list[4]
            df_patient[i] = df_list1[4]
            
        elif Idx_list[i] == 3:
            age[i] = 60
            df_m3[i] = df_list[3]
            df_patient[i] = df_list1[3]

        elif Idx_list[i] == 2:
            age[i] = 50
            df_m3[i] = df_list[2]
            df_patient[i] = df_list1[2]

        elif Idx_list[i] == 1:
            age[i] = 40
            df_m3[i] = df_list[1]
            df_patient[i] = df_list1[1]

        else:
            age[i] = 0
            df_m3[i] = df_list[0]
            df_patient[i] = df_list1[0]

    tts = []
    for Idx in range(len(Idx_list)):        
        ModelName.append(MachineLearning.ModelName[Idx])
        Accuracy.append(MachineLearning.Accuracy[Idx])
        FOneScore.append(MachineLearning.FOneScore[Idx])
        Precision.append(MachineLearning.Precision[Idx])
        Recall.append(MachineLearning.Recall[Idx])
        Train_Test_Split.append(MachineLearning.Train_Test_Split[Idx])
        Age.append(age[Idx])
        
        #print("TrainTest Split : ",MachineLearning.Train_Test_Split[Idx])    
        t1 = MachineLearning.Train_Test_Split[Idx].split("&")
        t2 = t1[1].split(":")
        t3 = t2[1].strip()[:-1]
        #print("\n*Test Percent : ",(float(t3)/100),'\n\n')
        tts.append(float(t3)/100)    
    
    for n in range(len(df_patient)):
        df = pd.read_csv('../Datasets/finalDataset_'+str(age[n])+'_patient.csv')
        print('../Datasets/finalDataset_'+str(age[n])+'_patient.csv')
        df, label = MachineLearning.Preprocessing(df_patient[n])
        
        parameter_list[n]["train_test_split"] = [tts[n]]
        parameter_list[n]["model_file"] = parameter_list[n]["model_file"]+"_p"
        MachineLearning.Train_Model(df, label, parameter_list[n])
    
    ModelName_P = MachineLearning.ModelName_P
    Accuracy_P = MachineLearning.Accuracy_P
    FOneScore_P = MachineLearning.FOneScore_P
    Precision_P = MachineLearning.Precision_P
    Recall_P = MachineLearning.Recall_P
    Train_Test_Split_P = MachineLearning.Train_Test_Split_P
    Age_P = age
    #Writing
    f2 = open('../BackupResults/stats_main_p.dat', 'w')
    f1 = open('../BackupResults/stats_main.dat', 'w')
    for n in range(len(ModelName)):
        line1 = ModelName[n]+","+Train_Test_Split[n]+","+str(Accuracy[n])+","+str(FOneScore[n])+","+str(Precision[n])+","+str(Recall[n])+","+str(Age[n])+"*"
        line2 = ModelName_P[n]+","+Train_Test_Split_P[n]+","+str(Accuracy_P[n])+","+str(FOneScore_P[n])+","+str(Precision_P[n])+","+str(Recall_P[n])+","+str(Age_P[n])+"*"
        f1.write(line1)
        f2.write(line2)
    f1.close()
    f2.close()


    print('$'*5,'='*20,'#'*5,'='*20,'$'*5)
    print('$'*5,' Display Result ','$'*5)
    print('-'*50)

    choice = 'y'
    print('$'*5,'='*20,'#'*5,'='*20,'$'*5)
    print('$'*5,' Display Graphical Analysis Result ','$'*5)
    while choice in ['y','Y']:
        print('$'*5,'Analysis of models using graphs','$'*5)
        print(' 1) Model Comaprison on Datasets replacing NAN to Ages as 0,40,50,60,70 \n 2) ML Models Comparison \n 3) Model Comparison on finalDataset and patientDataset')
        print('-'*50)
        var = sys.maxsize
        while var > 3:
            try:
                var = int(input("Choice? (1/2/3) "))
                if var > 3:
                    print('Invalid Selection. Please select Again')
                elif var < 1:
                    print('Invalid Selection. Please select Again')
                    var = sys.maxsize
            except:
                    print('Please choose from given options')  
                    var = sys.maxsize 
        if var == 1:  
            DisplayGraphs.Bar_Graph_0()
        elif var == 2:
            DisplayGraphs.Bar_Graph_1()
        else:
            DisplayGraphs.Bar_Graph_3()
        try:
            choice = input('Do you want to view more analysis graphical result? (y/n) ')
        except:
            print('Invalid Input!! Exiting')

    print('$'*5,'='*20,'#'*5,'='*20,'$'*5)
    print('$'*5,' Display Tabular Result ','$'*5)
    DisplayResult.PrintResult_0()
    print('\t','='*5,'-'*20,'  AGE : ',age,'  ','-'*20,'='*5)    
    DisplayResult.PrintResult_1()
    
    print('\n\n\n')
    print('\t $$ AVERAGE ACCURACY (merged.csv)             : ', sum(Accuracy)/len(Accuracy))
    print('\t $$ AVERAGE ACCURACY (patientDataset.csv) : ', sum(Accuracy_P)/len(Accuracy_P))
    print('\n\n\n')

    choice = input('Do you want to test model on user input?(y/n)   ')
    while choice in ['Y','y']:
        #label .again
        var = sys.maxsize    
        print('-'*50)
        print('$'*5,'Test Machine Learning Model on User Inputs','$'*5)
        print(' 1) Test ML Model based on Patient Dataset\n 2) Test ML Model based on Merged Dataset\n')
        print('-'*50)
        while var > 2:
            try:
                var = int(input('Enter Test Type : '))
                if var > 2:
                    print('Invalid Selection. Please select Again')
                elif var < 1:
                    print('Invalid Selection. Please select Again')
                    var = sys.maxsize
            except:
                print('Please choose from given options')
        if var == 1:
            MachineLearning.Test_model(True)

        elif var == 2:
            MachineLearning.Test_model(False)
            
        var = sys.maxsize
        try:
            choice = input('Want to try other test dataset?(y/n)   ')
        except:
            print('Incorrect Input, Exiting from user testing')
    else:
        print('Testing model on user input skipped!')
    
    choice = input('Do you want to print predictions & results?(y/n)   ')
    if choice == 'y' or choice == 'Y':
        DisplayResult.PrintResult_2()
    else:
        print('Result skipped!')
        
    choice = input('Do you want to perform cleanup routine?(y/n)   ')
    if choice == 'y' or choice == 'Y':
        Cleanup.Cleanup_Routine()
    else:
        print('Cleanup routine skipped!')
