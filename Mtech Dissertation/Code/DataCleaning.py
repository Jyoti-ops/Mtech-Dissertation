import pandas as pd

def Merge_Datasets():
    print('\n','$'*5,'-'*20,'='*5,'-'*20,'$'*5)
    print('\t','$'*5,' Merging Dataset Operations ','$'*5)
    print('\n#Merging Datasets Region.csv and PatientInfo.csv on \'city\' attribute')
    data1 = pd.read_csv('../Datasets/Region.csv')
    data2 = pd.read_csv('../Datasets/PatientInfo.csv')

    # using merge function by setting how='outer'
    output4 = pd.merge(data1, data2,on='city',how='outer')
    print('#First merging has been completed and merged file has been saved as m1.csv in Dataset/ folder')
    # displaying result
    #print(output4)
    output4.to_csv('../Datasets/m1.csv')

    print('#Merging Datasets previously merged dataset m1.csv and Weather.csv on \'province_x\' & \'date\' attribute')
    data1 = pd.read_csv('../Datasets/m1.csv')
    data2 = pd.read_csv('../Datasets/Weather.csv')

    # using merge function by setting how='outer'
    output4 = pd.merge(data1, data2,on=['province_x','date'],how='outer')
    
    # displaying result
    #print(output4)
    output4.to_csv('../Datasets/m2.csv')
    print('#Second merging has been completed and merged file has been saved as m2.csv in Dataset/ folder')

def AgeDatasetOperation(df, age, dataset):
    #print('NAN Values for ',dataset,'_',age,'.csv : ',df['age'].isna().sum())
    df['age']=df['age'].fillna(age)
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    try:
        df.drop(['Unnamed: 0'],axis=1,inplace=True)
        df.drop([''],axis=1,inplace=True)
    except:
        pass
    if dataset == "PatientInfo":
        df.to_csv('../Datasets/finalDataset_'+str(int(age))+'_patient.csv')
    else:    
        df.to_csv('../Datasets/finalDataset_'+str(int(age))+'.csv')
    
def Clean_Dataset(dataset='m3.csv'):
    print('\n','$'*5,'-'*20,'='*5,'-'*20,'$'*5)
    print('\t','$'*5,' Cleaning Dataset Operations ','$'*5)
    
    print('\n#Cleaning dataset PatientInfo.csv')
    # df = pd.read_csv('../Datasets/PatientInfo.csv')
    # df.drop(['date','released_date','id','country'],axis=1,inplace=True)    
    # df['age'] = df['age'].str.replace('s', '')
    # df['age'] = df['age'].astype(float, errors = 'raise')
    # print(df)
    age_list = [0.0,40.0,50.0,60.0,70.0]
    for age in age_list:
        df = pd.read_csv('../Datasets/PatientInfo.csv')
        df.drop(['date','released_date','id','country'],axis=1,inplace=True)    
        df['age'] = df['age'].str.replace('s', '')
        df['age'] = df['age'].astype(float, errors = 'raise')
        AgeDatasetOperation(df, age, "PatientInfo")
    
    df = pd.read_csv('../Datasets/m2.csv')

    print('\n#Reading previously merged dataset m2.csv and converting \'Series Data\' to \'Categorical Data\'')
    df['avg_temp_group']=pd.cut(
        df['avg_temp'], 
        bins=[-10, 10, 20, 30],
        labels=[ 'cold', 'Normal', 'hot']
    )

    df['precipitation_group']=pd.cut(
        df['precipitation'], 
        bins=[-1,50,300],
        labels=['L','H']
    )

    df['max_wind_speed_group']=pd.cut(
        df['max_wind_speed'], 
        bins=[0,6,12,18], 
        labels=['low','avg','strong']
    )

    df['most_wind_direction_group']=pd.cut(
        df['most_wind_direction'], 
        bins=[0,80,90,170,180,260,270,350,360], 
        labels=['NE','E','SE','S','SW','W','NW','N']
    )

    df['avg_relative_humidity_group']=pd.cut(
        df['avg_relative_humidity'], 
        bins=[0,30,60,100], 
        labels=['low','normal','high']
    )

    print('#Converting \'Series Data\' to \'Categorical Data\' had been completed Successfully and has been saved as m3.csv in Dataset/ folder')

    df.drop(['code_x','code_y','date','released_date','Unnamed: 0','Unnamed: 0.1','id','country','province_y','avg_temp','min_temp','max_temp','precipitation','max_wind_speed','most_wind_direction','avg_relative_humidity'],axis=1,inplace=True)
    df['age'] = df['age'].str.replace('s', '')
    df['age'] = df['age'].astype(float, errors = 'raise')

    df.rename(columns = {('province_x'):('province')}, inplace=True)
    df.to_csv('../Datasets/m3.csv')

    age_list = [0.0,40.0,50.0,60.0,70.0]
    for age in age_list:
        df = pd.read_csv('../Datasets/m3.csv') # renamed from the csv file within train.csv.zip on Kaggle
        #df['date'] = df['date'].str.replace('/', '-')
        AgeDatasetOperation(df, age, "m3")   