import prettytable
import MachineLearning as ml

def PrintResult_0():
    my_table = prettytable.PrettyTable()
    my_table.field_names = ["Model", "Train Test Split", "Accuracy", "F1 Score","Precision","Recall"]
    if len(ml.Accuracy) == 0:
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
            my_table.add_row([data[0], data[1], data[2]+"%", data[3]+"%", data[4]+"%", data[5]+"%"])
            ml.ModelName.append(data[0])            
            ml.Train_Test_Split.append(data[1])
            ml.Accuracy.append(int(data[2]*100))
            ml.FOneScore.append(int(data[3]*100))
            ml.Precision.append(int(data[4]*100))
            ml.Recall.append(int(data[5]*100))
            #print(Recall)
            
    else:
        for index in range(len(ml.ModelName)):
            my_table.add_row([ml.ModelName[index], ml.Train_Test_Split[index] , str(ml.Accuracy[index])+'% ', str(ml.FOneScore[index])+'% ', str(ml.Precision[index])+'% ' , str(ml.Recall[index])+'% '])
            if index in [5,11,17,23]:
                my_table.add_row(['  ------  ', '  ------  ' , '  ------  ', '  ------  ', '  ------  ' , '  ------  '])

    #Person
    # my_table_p = prettytable.PrettyTable()
    # my_table_p.field_names = ["Model", "Train Test Split", "Accuracy", "F1 Score","Precision","Recall"]

    # if len(ml.Accuracy_P) == 0:
    #     with open('stats_p.dat', 'r') as f:
    #         data1 = f.readlines()
    #     f.close()
    #     data2 = data1[0].split('*')
    #     #print("DATA1 : ",data2)
    #     for data3 in data2:
    #         if data3 == '' or data3 == None:
    #             break
    #         data = data3.split(',')
    #         #print("\n\n\nDATA : ",data,"\n\n\n")
    #         my_table_p.add_row([data[0], data[1], data[2]+"%", data[3]+"%", data[4]+"%", data[5]+"%"])
    #         ml.ModelName_P.append(data[0])            
    #         ml.Train_Test_Split_P.append(data[1])
    #         ml.Accuracy_P.append(int(data[2]*100))
    #         ml.FOneScore_P.append(int(data[3]*100))
    #         ml.Precision_P.append(int(data[4]*100))
    #         ml.Recall_P.append(int(data[5]*100))
    #         #print(Recall)
            
    # else:
    #     for index in range(len(ml.ModelName_P)):
    #         my_table_p.add_row([ml.ModelName_P[index], ml.Train_Test_Split_P[index] , str(ml.Accuracy_P[index])+'% ', str(ml.FOneScore_P[index])+'% ', str(ml.Precision_P[index])+'% ' , str(ml.Recall_P[index])+'% '])

    print('\n\n\n')
    print('\n\t','-'*20,'$'*5,'Dataset(finalDataset.csv)','$'*5,'-'*20,)
    print(my_table)
    # print('\n\t','-'*20,'$'*5,'Patient Dataset(patientDataset.csv)','$'*5,'-'*20,)
    # print(my_table_p)
    print('\n\n\n')
    

def PrintResult_1():
    my_table = prettytable.PrettyTable()
    my_table.field_names = ["Model", "Train Test Split", "Accuracy", "F1 Score","Precision","Recall", "Age"]

    with open('../BackupResults/stats_main.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split(',')
        my_table.add_row([data[0], data[1], data[2]+"%", data[3]+"%", data[4]+"%", data[5]+"%", data[6]])

    #Person
    my_table_p = prettytable.PrettyTable()
    my_table_p.field_names = ["Model", "Train Test Split", "Accuracy", "F1 Score","Precision","Recall", "Age"]

    with open('../BackupResults/stats_main_p.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split(',')
        #print("\n\n\nDATA : ",data,"\n\n\n")
        my_table_p.add_row([data[0], data[1], data[2]+"%", data[3]+"%", data[4]+"%", data[5]+"%", data[6]])
                
    print('\n\n')
    print('\n\t','-'*20,'$'*5,'Merged Dataset(finalDataset.csv)','$'*5,'-'*20,)
    print(my_table)
    print('\n\t','-'*20,'$'*5,'Patient Dataset(patientDataset.csv)','$'*5,'-'*20,)
    print(my_table_p)
    print('\n\n')
    

def PrintResult_2():
    my_table = prettytable.PrettyTable()
    my_table.field_names = ["Model", "Input", "Output(Risk)"]

    with open('../BackupResults/test_results.dat', 'r') as f:
        data1 = f.readlines()
    f.close()
    data2 = data1[0].split('*')
    #print("DATA1 : ",data2)
    for data3 in data2:
        if data3 == '' or data3 == None:
            break
        data = data3.split('$')
        my_table.add_row([data[0], data[1], data[2]])
                        
    print('\n\n')
    print('\n\t','-'*20,'$'*5,'Predicted Results','$'*5,'-'*20,)
    print(my_table)
    print('\n\n')


