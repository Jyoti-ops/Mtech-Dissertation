from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
import pickle
import os.path
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
import random as r
import pandas as pd

#Global Variables
ModelName, Train_Test_Split, Accuracy, FOneScore, Precision, Recall = [],[],[],[],[],[]
ModelName_P, Train_Test_Split_P, Accuracy_P, FOneScore_P, Precision_P, Recall_P = [],[],[],[],[],[]
label = "state"
#Check if stats.dat exist
def recoverBackup():
    if len(Accuracy) == 0:
        with open('../BackupResults/stats.dat', 'r') as f:
            data1 = f.readlines()
        f.close()
        data2 = data1[0].split('*')
        #print("DATA1 : ",data2)
        for data3 in data2:
            if data3 == '' or data3 == None:
                break
            data = data3.split(',')
            #print("\n\n\nDATA : ",data,"\n\n\n")
            ModelName.append(data[0])            
            Train_Test_Split.append(data[1])
            Accuracy.append(int(data[2]))
            FOneScore.append(int(data[3]))
            Precision.append(int(data[4]))
            Recall.append(int(data[5]))
                #print(Recall)

    if len(Accuracy_P) == 0:
        with open('../BackupResults/stats_p.dat', 'r') as f:
            data1 = f.readlines()
        f.close()
        data2 = data1[0].split('*')
        #print("DATA1 : ",data2)
        for data3 in data2:
            if data3 == '' or data3 == None:
                break
            data = data3.split(',')
            #print("\n\n\nDATA : ",data,"\n\n\n")
            ModelName_P.append(data[0])            
            Train_Test_Split_P.append(data[1])
            Accuracy_P.append(int(data[2]))
            FOneScore_P.append(int(data[3]))
            Precision_P.append(int(data[4]))
            Recall_P.append(int(data[5]))
            #print(Recall)

def Preprocessing(df):
    numerical_values = []
    categorical_values = []
    #counter = 0
    #Need Only 1 dataset
    
    for col in df.columns:
        dict = {}
        if col not in ["state"]:
            f = open("../LabelMaps/"+col+".dat", "w")
        else:
            f = open("../"+col+".dat", "w")
        le = preprocessing.LabelEncoder()
        le.fit(df[col].tolist())
        numerical_values.append(le.transform(list(le.classes_)))
        categorical_values.append(list(le.inverse_transform(numerical_values[-1])))
        for i in range(len(numerical_values[-1])):
            dict[categorical_values[-1][i]] = numerical_values[-1][i]
        df[col] = df[col].map(dict)
        f.write(str(dict))
        f.close()
    global label
    try:
        label1 = df.pop('state')
        label = label1
        #print("\nType : ",type(df.pop("city")))
    except:
        label1 = label
    #df.pop('srno')    
    return df, label1

def Train_Model(df, label, model={"model_name": "Random Forest Classifier", "model_file": "rfc","train_test_split":[0.20,0.25,0.3,0.35,0.4,0.45,0.5]}):
    if os.path.exists("../TrainedModels/"+str(model['model_file'])+"_trained_model.pickle"):
        print("Loading Trained Random Forest Classifier Model")
        classifier = pickle.load(open("../TrainedModels/"+str(model['model_file'])+"_trained_model.pickle", "rb"))
    else:
        #print("\n\n# DF.COLUMNS.COUNT : ", len(df.columns),'\n\n')
        print(f'Creating and training new '+str(model['model_file'])+' model')
        print('$ Training Started $')
        print()
        print(' -- $ ACCURACY ('+model['model_name']+') $ --')
        best_accuracy, ttn, best_precision, best_recall, best_f1 = 0,0,0,0,0
        for n in model["train_test_split"]:        
            X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=n, shuffle=True)
            if model['model_name'] == "Random Forest Classifier":
                classifier = RandomForestClassifier() 
            elif model['model_name'] == "K-Nearest Neighbour Classifier":
                classifier = KNeighborsClassifier(n_neighbors=5)
            elif model['model_name'] == "Naive Bayes Classifier":
                classifier = GaussianNB()
            elif model['model_name'] == "Support Vector Machine":
                classifier = SVC()
            elif model['model_name'] == "Neural Networks":
                classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
            else:
                classifier = LogisticRegression()

            classifier.fit(X_train.values, y_train.values)
            y_pred = classifier.predict(X_test.values) 
            # performing predictions on the test dataset
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred, average='macro')
            recall = metrics.recall_score(y_test, y_pred, average='macro')
            f1 = metrics.f1_score(y_test, y_pred, average='macro')

            print("Testing Dataset (",n*100,"%) : ", metrics.accuracy_score(y_test, y_pred))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_precision = precision
                best_recall = recall
                best_f1 = f1
                with open('../TrainedModels/'+str(model['model_file'])+'_trained_model.pickle', "wb") as file:
                    pickle.dump(classifier, file)
                ttn = n

        #print('\n\n**TRAIN_MODEL(Testing p) : ',model['model_file'][-1],'\n\n')
        if model['model_file'][-1] == 'p':
            global Train_Test_Split_P,Accuracy_P, FOneScore_P, Precision_P, Recall_P, ModelName_P
            Train_Test_Split_P.append("Train:"+str(100-ttn*100)+"% & Test:"+str(ttn*100)+"%")
            Accuracy_P.append(int(best_accuracy*100))
            FOneScore_P.append(int(best_f1*100))
            Precision_P.append(int(best_precision*100))
            Recall_P.append(int(best_recall*100))
            ModelName_P.append(model['model_name'])
            
            f = open('../BackupResults/stats_p.dat', 'a')
            line = ModelName_P[-1]+","+Train_Test_Split_P[-1]+","+str(Accuracy_P[-1])+","+str(FOneScore_P[-1])+","+str(Precision_P[-1])+","+str(Recall_P[-1])+"*"
        else:
            global Train_Test_Split,Accuracy, FOneScore, Precision, Recall, ModelName
            Train_Test_Split.append("Train:"+str(100-ttn*100)+"% & Test:"+str(ttn*100)+"%")
            Accuracy.append(int(best_accuracy*100))
            FOneScore.append(int(best_f1*100))
            Precision.append(int(best_precision*100))
            Recall.append(int(best_recall*100))
            ModelName.append(model['model_name'])
            
            f = open('../BackupResults/stats.dat', 'a')
            line = ModelName[-1]+","+Train_Test_Split[-1]+","+str(Accuracy[-1])+","+str(FOneScore[-1])+","+str(Precision[-1])+","+str(Recall[-1])+"*"
        #line = model['model_name']+" ,"+"Train:"+str(100-ttn*100)+"%, Test:"+str(ttn*100)+"% ,"+str(best_accuracy)+", "+str(best_f1)+", "+str(best_precision)+", "+str(best_recall)
        f.write(line)
        f.close()
        
        print()
        
        print('$ Training Completed $')
    print('*'*20)

def Test_model(isPatient):
    #Read File
    col_list = []
    input_string = ['']
    input_int = [0]
    if isPatient:
        df = pd.read_csv('../Datasets/finalDataset_0_patient.csv')
    else:
        df = pd.read_csv('../Datasets/finalDataset_0.csv')
    
    #For city
    province = ""
    attribute_list = df.columns.tolist()[1:]  #[1:] Remove Unnamed From Dataset
    attribute_list.remove("state")
    for a in attribute_list:  
        dict = {}       
        str_value = df[a].unique().tolist() 
        str_value.sort()  
        for key, value in enumerate(str_value):
            dict[key] = value
        col_list.append(dict)
        
    # print("COLUMN SIZE : ",len(col_list))
    # print("COLUMN_LIST : ",col_list)
    i = 0
    for records in col_list:
        record_list = []
        #city = -1
        #print("\n\n#Province : ",province)
        if province != "":
            str1_list = []
            #city = 0
            city_list = df['city'].unique().tolist();
            #print('\nCity List : ',city_list)
            for index, rows in df.iterrows():                
                for city in city_list:
                    if rows['province'] == province and rows['city'] not in str1_list:
                        #print("rows['province'] : ",rows['province'],"  rows['city'] : ",rows['city'])
                        str1_list.append(city)                        

        for key in records:
            if province == "":
                print(key,") ",records[key])
            elif records[key] in str1_list:
                print(key,") ",records[key])
            record_list.append(records[key])
        var = sys.maxsize
        while var > len(records)-1:
            try:
                var = int(input(str(attribute_list[i])+' : '))
                if var > len(records)-1:
                    print('Invalid Selection. Please select Again')
                elif var < 0:
                    print('Invalid Selection. Please select Again')
                    var = sys.maxsize
            except:
                print('Please input a number from above options')   
        i = i+1
        #Code for Getting selective city
        try:
            if "Jeollabuk-do" in record_list:
                province = records[var]   
            if "Hwaseong-si" in record_list:
                province = ""         
        except:
            pass
        input_string.append(records[var])
        input_int.append(var)
        
    record = [(input_string[i], input_int[i]) for i in range(1, len(input_int))]
    
    print('\n','--*'*6)
    print(record)
    print('--*'*6,'\n\n')

    #Model selection to test
    print('-'*50)
    print('$'*5,'Machine Learning Models','$'*5)
    print(' 1) Random Forest Classifier\n 2) K-Nearest Neighbour Classifier\n 3) Support Vector Machine Classifier\n 4) Naive Bays Classifier\n 5) Neural Network\n 6) Logistic Regression\n')
    print('-'*50)
    var = sys.maxsize
    choice = 'y'
    
    f = open('../BackupResults/test_results.dat', 'a')
    
    while choice in ['y', 'Y']:
        while var > 6:
            try:
                var = int(input('Please Choose Machine Learning Model to Test : '))
                if var > 6:
                    print('Invalid Selection. Please select Again')
                elif var < 1:
                    print('Invalid Selection. Please select Again')
                    var = sys.maxsize
            except:
                print('Please input a number from above options')
        
        if var == 1:
            if isPatient:
                fpath = "../TrainedModels/rfc 1234_p_trained_model.pickle"
            else:
                fpath = "../TrainedModels/rfc _trained_model.pickle"
            fs = "rfc "
            f.write("Random Forest Classifier$")
            if not os.path.exists(fpath):
                Train_Model(df, label, {"model_name": 'Random Forest Classifier', "model_file": fs})
            model = pickle.load(open(fpath, "rb"))

        elif var == 2:
            if isPatient:
                fpath = "../TrainedModels/knn 1234_p_trained_model.pickle"
            else:
                fpath = "../TrainedModels/knn _trained_model.pickle"
            fs = "knn "
            f.write("K-Nearest Neighbour Classifier$")
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'K-Nearest Neighbour Classifier', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))
            
        elif var == 3:
            if isPatient:
                fpath = "../TrainedModels/svm 1234_p_trained_model.pickle"
            else:
                fpath = "../TrainedModels/svm _trained_model.pickle"
            fs = "svm "
            f.write("Support Vector Machine$")
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Support Vector Machine', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        elif var == 4:
            if isPatient:
                fpath = "../TrainedModels/nb 1234_p_trained_model.pickle"
            else:
                fpath = "../TrainedModels/nb _trained_model.pickle"
            fs = "nb "
            f.write("Naive Bayes Classifier$")
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Naive Bayes Classifier', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        elif var == 5:
            if isPatient:
                fpath = "../TrainedModels/nn 1234_p_trained_model.pickle"
            else:
                fpath = "../TrainedModels/nn _trained_model.pickle"
            fs = "nn " 
            f.write("Neural Networks$")
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Neural Networks', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        else:
            if isPatient:
                fpath = "../TrainedModels/lr 1234_p_trained_model.pickle"
            else:
                fpath = "../TrainedModels/lr _trained_model.pickle"
            fs = "lr "
            f.write("Logistic Regression$")
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Logistic Regression', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        #print('\n\n\nMODEL : ',fpath,'\n\n\n')
        #print("\n\n$Input ",input_str," : ")
        
        
        f.write(str(input_string[1:])+"$")
        
        res =  model.predict(np.array([input_int]))
        if res == 0:
            print('*Predicted Output : RISK - VERY HIGH')
            f.write("VERY HIGH$")
        elif res == 1:
            print('*Predicted Output : RISK - HIGH/MODERATE')
            f.write("MODERATE$")
        elif res == 2:
           print('*Predicted Output : RISK - LOW')
           f.write("LOW$")
        else:
            print(res)
            
        f.write("*")
        
        print('\n\n\n')
        var = sys.maxsize
        try:
            choice = input('Want to test same input on other model? (y/n)   ')
        except:
            print('Invalid Input! Exiting!')
            choice = 'n'           
    f.close()
        
        
        
                    
def Test_model1(df, label):
    print('NOTE: Please select attributes from list')
    #Why 6? there are 6 attributes in patientDataset.csv
    lst_file = []
    patient = ''
    for attributes in df.columns:
        if attributes != "Unnamed: 0":
            lst_file.append(str(attributes)+".txt")
    #print('\n\n\n\n####lst_file : ',lst_file,'\n\n\n\n')
    if 'avg_temp_group.txt' not in lst_file:
        #print('\'avg_temp_group.txt\' in lst_file')
        patient = '_p'

    list1 = []
    var = sys.maxsize
    input_str = []
    input_num = []
    mydict1 = {}
    for file in lst_file:
        with open("../LabelMaps/"+file) as f:
            lines = f.readlines()
        a = lines[0].split(",")
        dict = {}
        for item in a:
            b = item.split(":")
            b[0] = b[0].replace('{', '')
            b[1] = b[1].replace('}', '')
            if not b[0].replace('.','',1).isdigit():
                b[0] = b[0].replace('\'','')
                b[0] = b[0].strip()
            dict[b[0]] = int(b[1])
        list1.append(dict)

    i,j = 0,0 
    #Menu driven Program and store result
    ip_province = -1
    for records in list1:
        #Considering fix column for city
        '''
        if ip_province == -1 and "Jongno-gu" in records:
           list1.append(records) 
        elif "Jongno-gu" in records:
            print("Input Province : ",ip_province)
            listt = []
            df = df.reset_index()
            key_list = list(records.keys())
            val_list = list(records.values())
            print("\n\n\n\n## RECORDS : ",records,'\n## KEY LIST : ',key_list,'\n\n\n\n\n')
            
            # rec_list = []
            # rec_id = []
            # for rec in key_list:
            #     if rec not in rec_list:
            #         rec_list.append(rec)
            #         rec_id.append()
            for index, rows in df.iterrows():
                
                print("rows['province']  : ",rows['province'],end=", ")
                
                # if ip_province > len(key_list):
                #     ip_province = r.randint(len(key_list))
                if rows['province'] == ip_province:
                    print('\n\n#Rows : ',rows) #,'\nKey List : ',key_list[rows['city']])
                    
                    if key_list[rows['city']] not in listt:
                        try:
                            print(rows['city'],") ",key_list[rows['city']])
                            # position = val_list.index(rows['city'])
                            # print(rows['city'],") ",key_list[position]+1)
                            listt.append(rows['city'])
                        except:
                            listt.append(key_list[r.randint(0,6)])
            print("Listt length : ",len(listt))
            # for rec in listt:
            #     print(j+1,") ",rec)
            #     j = j+1
            var = sys.maxsize
            try:
                var = int(input(str(lst_file[i].replace('.txt', ' '))+' : '))
                if var not in listt:
                    print('Invalid Selection. Please select Again, Selecting Idx 1 as default')
                    var = 1
            except:
                print('Invalid Selection. Please select Again, Selecting Idx 1 as default')
                var = 1                
        else:
            '''
        for rec in records:
            print(j+1,") ",rec)
            j = j+1
        var = sys.maxsize
        while var > len(records):
            try:
                var = int(input(str(lst_file[i].replace('.txt', ' '))+' : '))
                if var > len(records):
                    print('Invalid Selection. Please select Again')
                elif var < 1:
                    print('Invalid Selection. Please select Again')
                    var = sys.maxsize
            except:
                print('Please input a number from above options')
        if "Daegu" in records:
            print("\n\n\n\n## PROVINCE RECORDS : ",records,'\n\n\n\n\n')
            # key_list = list(records.keys())
            # val_list = list(records.values())
            # position = val_list.index(var-1)
            #print(key_list[position])
            ip_province = var-1
            
        #print(records)
        # list = [1,2,3,4,6]
        # list[2]
        # records = {'male':0, "female":1}
        # list[records.keys()] = ['male','female'][1] = 'female'
        # list(records.values()) = [0,1]
        # mydict[['male','female']][1] = 1
        # mydict = {'female':1, '30.0':2, 'Seol':15}
        mydict1[list(records.keys())[list(records.values()).index(var-1)]] = var-1
        input_num.append(var-1)
        input_str.append(list(records.keys())[list(records.values()).index(var-1)])
        i = i+1
        j = 0
        var = sys.maxsize
    input_num = []
    input_str = []
    files1 = [f.replace('.txt','') for f in lst_file]
    temp = list(mydict1)
    #print('temp : ',temp,'\nmydict1 : ',mydict1,'\ncol_sq : ',col_sq)
    for columns in df.columns:
        #print("mydict1[temp[files1.index(columns)]] : ",mydict1[temp[files1.index(columns)]],"\ntemp[files1.index(columns)] : ",temp[files1.index(columns)],"\nfiles1.index(columns) : ",files1.index(columns),"\n")
        input_num.append(mydict1[temp[files1.index(columns)]])
        input_str.append(temp[files1.index(columns)])
    record = [(input_str[i], input_num[i]) for i in range(0, len(input_num))]
    print('\n','--*'*6)
    print(record)
    print('--*'*6,'\n\n')

    #Model selection to test
    print('-'*50)
    print('$'*5,'Machine Learning Models','$'*5)
    print(' 1) Random Forest Classifier\n 2) K-Nearest Neighbour Classifier\n 3) Support Vector Machine Classifier\n 4) Naive Bays Classifier\n 5) Neural Network\n 6) Logistic Regression\n')
    print('-'*50)
    var = sys.maxsize
    choice = 1000
    while choice != 0:
        while var > 6:
            try:
                var = int(input('Please Choose Machine Learning Model to Test : '))
                if var > 6:
                    print('Invalid Selection. Please select Again')
                elif var < 1:
                    print('Invalid Selection. Please select Again')
                    var = sys.maxsize
            except:
                print('Please input a number from above options')
        
        if var == 1:
            fpath = "../TrainedModels/rfc"+str(patient)+"_trained_model.pickle"
            fs = "rfc"+str(patient)
            if not os.path.exists(fpath):
                Train_Model(df, label, {"model_name": 'Random Forest Classifier', "model_file": fs})
            model = pickle.load(open(fpath, "rb"))

        elif var == 2:
            fpath = "../TrainedModels/knn"+str(patient)+"_trained_model.pickle"
            fs = "knn"+str(patient)
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'K-Nearest Neighbour Classifier', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))
            
        elif var == 3:
            fpath = "../TrainedModels/svm"+str(patient)+"_trained_model.pickle"
            fs = "svm"+str(patient)
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Support Vector Machine', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        elif var == 4:
            fpath = "../TrainedModels/nb"+str(patient)+"_trained_model.pickle"
            fs = "nb"+str(patient)
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Naive Bayes Classifier', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        elif var == 5:
            fpath = "../TrainedModels/nn"+str(patient)+"_trained_model.pickle"
            fs = "nn"+str(patient)
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Neural Networks', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        else:
            fpath = "../TrainedModels/lr"+str(patient)+"_trained_model.pickle"
            fs = "lr"+str(patient)
            if not os.path.exists(fpath):
                Train_Model(df, label, {'model_name': 'Logistic Regression', 'model_file': fs})
            model = pickle.load(open(fpath, "rb"))

        #print('\n\n\nMODEL : ',fpath,'\n\n\n')
        #print("\n\n$Input ",input_str," : ")
        res =  model.predict(np.array([input_num]))
        if res == 0:
            print('*Predicted Output : RISK - VERY HIGH')
        elif res == 1:
            print('*Predicted Output : RISK - HIGH/MODERATE')
        elif res == 2:
           print('*Predicted Output : RISK - LOW')
        else:
            print(res)
        print('\n\n\n')
        var = sys.maxsize
        try:
            choice = int(input('Want to test same input on other model? [Y/y = 1, N/n = 0]   '))
        except:
            choice = 1   