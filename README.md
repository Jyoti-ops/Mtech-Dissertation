# Mtech-Dissertation
Save Values and Repected IDs of  Dataset globally
1) **BEFORE EXECUTION, DELETE THE FILE INSIDE** **BackupResults, LabelMaps, TrainedModels and also delete all dataset EXCEPT PatientInfo.csv, Region.csv, Weather.csv inside Dataset Folder** 
2) After that execute Covid_Prediction.py within Code Folder.
3) why to delete?
    - So that you can train the model again otherwise it will show the result of existing trained output.
5) Setup Main Function
    - MergeDataset()
    - CleanDataset_All()
    - CleanDataset_Patient()
    - Preprocess_Dataframes()
    - TrainModel()
    - TestModel()
    - PrintResult() - Only Tables
    - PrintGraphs() - 
        -- BarGraph()
        -- PointGraph()
