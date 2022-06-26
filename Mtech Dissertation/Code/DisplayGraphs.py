import matplotlib.pyplot as plt
import numpy as np

rfc_list = list(range(0,30,6))
knn_list = list(range(1,30,6))
svm_list = list(range(2,30,6))
nb_list = list(range(3,30,6))
nn_list = list(range(4,30,6))
lr_list = list(range(5,30,6))


def Bar_Graph_0():
    print('Bar Graph')
    barWidth = 0.1
    Accuracy = []
    with open('../BackupResults/stats.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split(',')
        Accuracy.append(int(data[2]))

    RFC_A, KNN_A, SVM_A, NB_A, NN_A, LR_A = [],[],[],[],[],[]
    for n in rfc_list:
        RFC_A.append(Accuracy[n])
        
    for n in knn_list:
        KNN_A.append(Accuracy[n])
        
    for n in svm_list:
        SVM_A.append(Accuracy[n])
    
    for n in nb_list:
        NB_A.append(Accuracy[n])
        
    for n in nn_list:
        NN_A.append(Accuracy[n])
    
    for n in lr_list:
        LR_A.append(Accuracy[n])
        
    #fig = plt.subplots(figsize =(12, 8))
    #xticklabel = ModelName[:6]
    br1 = np.arange(6)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    #br6 = [x + barWidth for x in br5] 
       
    plt.bar(br1, Accuracy[:6], color ='r', width = barWidth, label ='Age = 0')      
    plt.bar(br2, Accuracy[6:12], color ='b', width = barWidth, label ='Age = 40')      
    plt.bar(br3, Accuracy[12:18], color ='g', width = barWidth, label ='Age = 50')      
    plt.bar(br4, Accuracy[18:24], color ='y', width = barWidth, label ='Age = 60')
    plt.bar(br5, Accuracy[24:30], color ='m', width = barWidth, label ='Age = 70')
    #plt.bar(br6, Accuracy[25:30], color ='k', width = barWidth, label ='Logistic Regression')
          
          
    plt.legend()
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Percentage")
    plt.xticks(np.arange(6),["RFC","KNN","SVM","NB","NN","LR"])
    plt.title("Machine Learning Model Comaprison")
    plt.show()
    
def Bar_Graph_1():
    print('Bar Graph')
    barWidth = 0.15
    ModelName, Accuracy, FOneScore, Precision, Recall = [],[],[],[],[]

    #fig = plt.subplots(figsize =(12, 8))
    #xticklabel = ModelName[:6]
    with open('../BackupResults/stats.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split(',')
        ModelName.append(data[0])
        Accuracy.append(int(data[2]))
        FOneScore.append(int(data[3]))
        Precision.append(int(data[4]))
        Recall.append(int(data[5]))    
    
    br1 = np.arange(6)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    
    plt.bar(br1, Accuracy[:6], color ='r', width = barWidth, label ='Accuracy')      
    plt.bar(br2, FOneScore[:6], color ='b', width = barWidth, label ='F1 Score')      
    plt.bar(br3, Precision[:6], color ='g', width = barWidth, label ='Precision')      
    plt.bar(br4, Recall[:6], color ='y', width = barWidth, label ='Recall')      
          
    plt.legend()
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Percentage")
    plt.xticks(np.arange(6),ModelName[:6])
    plt.title("Machine Learning Model Comaprison")
    plt.show()
    
def Bar_Graph_3():
    print('Bar Graph')
    barWidth = 0.3
    ModelName, Accuracy, FOneScore, Precision, Recall = [],[],[],[],[]
    ModelName_P, Accuracy_P, FOneScore_P, Precision_P, Recall_P = [],[],[],[],[]


    with open('../BackupResults/stats_main.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split(',')
        ModelName.append(data[0])
        Accuracy.append(int(data[2]))
        FOneScore.append(int(data[3]))
        Precision.append(int(data[4]))
        Recall.append(int(data[5]))    
    
    with open('../BackupResults/stats_main_p.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split(',')
        ModelName_P.append(data[0])
        Accuracy_P.append(int(data[2]))
        FOneScore_P.append(int(data[3]))
        Precision_P.append(int(data[4]))
        Recall_P.append(int(data[5]))
        
    br1 = np.arange(6)
    br2 = [x + barWidth for x in br1]
    #br3 = [x + barWidth for x in br2]
    #br4 = [x + barWidth for x in br3]
    
    plt.bar(br1, Accuracy, color ='r', width = barWidth, label ='Accuracy of merged.csv')
    plt.bar(br2, Accuracy_P, color ='g', width = barWidth, label ='Accuracy of patientDataset.csv')      
    # plt.bar(br3, Precision[:6], color ='g', width = barWidth, label ='Precision')      
    # plt.bar(br4, Recall[:6], color ='y', width = barWidth, label ='Recall')      
        
    plt.legend()
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Percentage")
    plt.xticks(np.arange(6),ModelName[:6])
    plt.title("Machine Learning Model Comaprison")
    plt.show()