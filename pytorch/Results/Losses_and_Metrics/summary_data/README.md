Here are two folders:

    1. epoch only
    2. step_included

    they all contain csv files and each file contains both DiceLoss and DiceAccuracy values in different columns
    but 'epoch only' means only the value at the end of each epoch is kept and the values recorded during training are disposed.
    On the other hand, 'step included' means all values are saved.
    This difference only affect the number of rows where the column 'Purpose' == train

There are 5 folders under each of the 'epoch only' and 'step_included' folder and their name represents the included models and trials:
    
    1. all : ([ResNet, DenseNet, Efficient] x [Pretrained, NonPretrained] + [AttDs])x[01-10 trials] + [Efficient Pretrained, AttDs] x [11-20 additional trails]
    2. 7_models_10_trails : ([ResNet, DenseNet, Efficient] x [Pretrained, NonPretrained] + [AttDs])x[01-10 trials]
    3. 6_models_10_trails : ([ResNet, DenseNet, Efficient] x [Pretrained, NonPretrained])x[01-10 trials]
    4. EfficientNet_vs_AttDs_20_trials : [Efficient Pretrained, AttDs] x [1-20 trails]
    5. EfficientNet_vs_AttDs_10_trials : [Efficient Pretrained, AttDs] x [1-10 trails]
        
All these 5 folders has 2 csv files:
    
    1. raw.csv : the raw data
    2. statistics.csv : aggregation of ['ModelName','IsPretrained','Purpose','Epoch'] of raw data and gives the statistics values of ['median','mean','std','count'] for all metrics
