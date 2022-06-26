import os

def Cleanup_Routine():
    print('\n\n\n#Executing Cleanup routine!!\n','-'*50)
    dir = ['..\\TrainedModels\\','..\\LabelMaps\\','..\\BackupResults\\']
    for directory in dir:
        for files in os.listdir(directory):
            os.remove(os.path.join(directory, files))
            print('File ',os.path.join(directory, files),' deleted successfully!!')

    # os.remove('stats.dat')
    # print('File stats.dat deleted successfully!!')
    # os.remove('stats_p.dat')
    # print('File stats_p.dat deleted successfully!!')
    dataset_dir = '..\\Datasets\\'
    filelist = [f for f in os.listdir(dataset_dir) if f.startswith("finalData") ]
    for file in filelist:
        os.remove(os.path.join(dataset_dir, file))
        print('File ',file,' deleted successfully!!')
        
    os.remove(os.path.join(dataset_dir, 'm1.csv'))
    print('File m1.csv deleted successfully!!')
    os.remove(os.path.join(dataset_dir, 'm2.csv'))
    print('File m2.csv deleted successfully!!')
    os.remove(os.path.join(dataset_dir, 'm3.csv'))
    print('File m3.csv deleted successfully!!')
    print('-'*50,'\n\n')