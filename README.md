# AIToolSoC
### In this project, four different algorithms are used for solving two different problems, about Li-On battery SoC and Discharge Current.
#### There is no claim that the created models work perfectly and are the best. The studies serve educational purposes.

------

The Algorithms used for the tasks are given below:

| Classification        | Prediction      |
| -------------         |:-------------:|
| KNN                   | Neural Network  |
| Decision Tree         | Linear Regression      |


The data is basically different discharge data of a Li-On battery cell, OC voltage against SoC.

The metrics are given below:

| -             | KNN           | Decision Tree   | Neural Network | Linear Regression | 
| ------------- |:-------------:|:---------------:|:--------------:|:-----------------:|
| mse           | 0.0           | 0.4725          |0.1273          | 0.0006224405147   |
| R2 Score      | 100%          | 99.31%          | -              | 91.247%           |
| Time Consumed | 0.0045422s    | 0.0079075s      |122.41s         | 0.0047117s        |

For neural networks, **Tensorflow 2** was used. For all other ones, **Scikit-Learn** classes used. 

The code is simple to understand, data obtaining parts are dependent for this spesific data only. 
