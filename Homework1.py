

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Importing the dataset

iris = pd.read_csv('Iriss.csv')
X = iris.iloc[:, 0:4].values
y = iris.iloc[:, 4].values
print(iris.head(5))

#QUESTION 1:  Converting Categorical data into Y to numerical data

from sklearn import preprocessing
for i in range(0,len(iris)):
    if iris['Class'][i]=='IrisSetosa':
        iris['Class'][i]=1
    elif iris['Class'][i]=='Iris-versicolor':
        iris['Class'][i]=2
    else:
        iris['Class'][i]=3
print (iris.head(5))

# QUESTION 2.1:  Creating 2D plots 

sns.pairplot(iris, x_vars=[ "SepalLength","SepalWidth",
                              "PetalLength","PetalWidth"], 
                      y_vars=["SepalLength","SepalWidth",
                              "PetalLength","PetalWidth"],
                      hue="Class", size=2, aspect=.8)


#QUESTION 2.2:  Create 3D plots

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
x1 =X[:,0]
x2 =X[:,1]
x3 =X[:,3]
ax.scatter(x1, x2, x3, c='b', marker='^')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Width')
plt.show()

#QUESTION 2.3:  Visualization of the feature matrix (column 1-4)

Array=X
image = plt.imshow(X, cmap='GnBu', aspect='auto',extent=[0,8,0,8], interpolation="none")
plt.colorbar(image, orientation='horizontal')
plt.show()

#QUESTION 2.4:  Histograms of the four attributes and 3 classes

#A.Plotting for Sepal Length

dataset3=pd.read_csv('Histogram.csv')
SLSetosa=dataset3.iloc[:, 0].values
SLVersicolor=dataset3.iloc[:, 1].values
SLViginica=dataset3.iloc[:, 2].values
from pylab import xticks
plt.hist(SLSetosa, alpha=1, label='Sepal Length Setosa',color = "skyblue", ec="blue")
plt.hist(SLVersicolor,alpha=1, label='Sepal Length Versicolor',color = "pink", ec="red")
plt.hist(SLViginica,alpha=1, label='Sepal Length Virginica',color = "lightyellow", ec="green")
xticks(range(0,8))
plt.legend(loc='upper left')
plt.show()

#B.Plotting for Sepal Width

SWSetosa=dataset3.iloc[:, 3].values
SWVersicolor=dataset3.iloc[:, 4].values
SWViginica=dataset3.iloc[:, 5].values
from pylab import xticks
bins = np.linspace(-10, 10, 100)
plt.hist(SWSetosa,alpha=1, label='Sepal Width Setosa',color = "skyblue", ec="blue")
plt.hist(SWVersicolor,alpha=1, label='Sepal Width Versicolor',color = "pink", ec="red")
plt.hist(SWViginica,alpha=1, label='Sepal Width Virginica',color = "lightyellow", ec="green")
xticks(range(1,5))
plt.legend(loc='upper left')
plt.show()

#C.Plotting for Petal Length

PLSetosa=dataset3.iloc[:, 6].values
PLVersicolor=dataset3.iloc[:, 7].values
PLViginica=dataset3.iloc[:, 8].values
from pylab import xticks
bins = np.linspace(-10, 10, 100)
plt.hist(PLSetosa,alpha=1, label='Petal Length Setosa',color = "skyblue", ec="blue")
plt.hist(PLVersicolor,alpha=1, label='Petal Length Versicolor',color = "pink", ec="red")
plt.hist(PLViginica,alpha=1, label='Petal Length Virginica',color = "lightyellow", ec="green")
xticks(range(1,10))
plt.legend(loc='upper center')
plt.show()

#D.Plotting for Petal Width

PWSetosa=dataset3.iloc[:, 9].values
PWVersicolor=dataset3.iloc[:, 10].values
PWViginica=dataset3.iloc[:, 11].values
from pylab import xticks
bins = np.linspace(-10, 10, 100)
plt.hist(PWSetosa,alpha=1, label='Petal Width Setosa',color = "skyblue", ec="blue")
plt.hist(PWVersicolor,alpha=1, label='Petal Width Versicolor',color = "pink", ec="red")
plt.hist(PWViginica,alpha=1, label='Petal Width Virginica',color = "lightyellow", ec="green")
xticks(range(0,3))
plt.legend(loc='upper center')
plt.show()

#QUESTION 2.5:  Boxplot of the four attributes for the three classes

boxplotdata=pd.read_csv('boxplot.csv')
sns.set(style="ticks")
sns.boxplot(x="TypeOfMeasure", y="Measure(cm)",hue="Class",data=boxplotdata,palette="PRGn");
sns.despine(offset=10, trim=True)

#QUESTION 2.6: Correlation Matrix

corr=iris.drop(['Class'], axis=1).corr(method='spearman')
print (corr)

#Correlation Matrix Visualization

plt.figure()
plt.imshow(corr, cmap='GnBu')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr)), corr.columns);
plt.suptitle('Correlation Matrix', fontsize=15, fontweight='bold')
plt.show()

#QUESTION 2.7: Parallel Coordinates

from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris, 'Class')
plt.show()

#QUESTION 3.1: Make a function for Minkowski Distance.  
#(Three function inputs: sample A, sample B, and distance order r)

'''CREATING MINKOWSKI DISTANCE FUNCTION'''

def minkowski_dist(a,b,r):
    subtract = abs(a-b)
    root = 1/r
    inner_power = np.power(subtract,r)
    total_sum = np.sum(inner_power)
    result = np.power(total_sum,root)
    return result




#QUESTION 3.2: Make a Matlab function for T-statistics Distance.  
#(Two function inputs: time series A, time series B)

'''CREATING FUNCTION FOR T-STATISTICS DISTANCE FORMULA'''

def tstat(a,b):
    subtract = np.abs(np.mean(a)-np.mean(b))
    sd=np.std(a-b)
    result=np.divide(subtract,sd)
    return result


#QUESTION 3.3: Make a Matlab function for Mahalanobis Distance. (Three function inputs: sample A, sample B, 
#and covariance matrix M.)
    
'''CREATING FUNCTION FOR MAHALANOBIS DISTANCE FORMULA'''

from numpy.linalg import inv
def maha(a,b,covariance_matrix):
    inverse = inv(covariance_matrix)
    subtract = a-b
    multi = np.dot(subtract,inverse)
    transposed = np.transpose(subtract)
    result = np.dot(multi,transposed)
    return result


#QUESTION 3.4: Assume a new iris sample S has a feature vector of [5.0000, 3.5000, 1.4600, 0.2540]. 
#Calculate the distances of the new sample to the 150 samples in the iris dataset. 
#using Minkowski distance with r = 1, 2, 100, respectively. Plot the obtained distances.
    
'''CALLING THE FUNCTION CREATED TO EVALUATE DISTANCES BETWEEN 
SAMPLE S AND 150 SAMPLES FOR r=1,2,100'''
distance_matrix = np.zeros([len(iris),len(iris)])
distance_matrix1 = np.zeros([len(iris),3])
r_tuple = [1,2,100]
b=np.asarray([5, 3.5,1.4,0.2540])
for i in range(0,len(r_tuple)):
    for j in range(0,len(iris)):
            a = np.asarray(iris.loc[j:j,'SepalLength':'PetalWidth'])
            print(a)
            r = r_tuple[i]
            print(r)
            value = minkowski_dist(a,b,r)
            distance_matrix1[j,i] = value
column_label = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
print (column_label)
print (distance_matrix1)

'''PLOTTING FOR r=1,2,100'''            
f1 = plt.figure(1)
plt.title("when r is 1")
plt.scatter(range(1,151),distance_matrix1[:,0])
f1.show()
f2 = plt.figure(2)
plt.title("when r is 2")
plt.scatter(range(1,151),distance_matrix1[:,1])
f2.show()
f3 = plt.figure(3)
plt.title("when r is 100")
plt.scatter(range(1,151),distance_matrix1[:,2])
f3.show()
plt.show()

#QUESTION 3.5: Assume a new iris sample S has a feature vector of [5.0000, 3.5000, 1.4600, 0.2540]. 
#Calculate the distances of the new sample to the 150 samples in the iris dataset.
#Using Mahalanobis distance. Plot the obtained distances. 

'''CALLING THE FUNCTION CREATED TO EVALUATE DISTANCES BETWEEN EACH OF THE 150 SAMPLES(150 BY 150 MATRIX)  '''

iris = iris.drop(['Class'], axis=1)
iris1 = iris.values
covariance_matrix=np.cov(iris1, rowvar=False)
print(covariance_matrix)
inverse = inv(covariance_matrix)
print(inverse)
distance_matrix = np.zeros([len(iris), len(iris)])
for i in range(0,len(iris)):
    for j in range(0,len(iris)):
        a = np.asarray(iris.loc[i:i,'SepalLength':'PetalWidth'])
        b = np.asarray(iris.loc[j:j,'SepalLength':'PetalWidth'])
        value = maha(a,b,covariance_matrix)
        distance_matrix[i,j] = value
print(distance_matrix)

'''CALLING THE FUNCTION CREATED TO EVALUATE DISTANCES BETWEEN SAMPLE S AND 150 
SAMPLES(150 BY 1 MATRIX)'''
distance_matrix1 = np.zeros([len(iris),1])       
b=np.asarray([5, 3.5,1.4,0.2540])
for i in range(0,len(iris)):
        a = np.asarray(iris.loc[i:i,'SepalLength':'PetalWidth'])
        value = maha(a,b,covariance_matrix)
        distance_matrix1[i] = value
        print(distance_matrix1)
        
'''PLOTTING THE DISTANCE MATRIX'''
f1 = plt.figure()
plt.title("Mahalanobis Distance ")
plt.scatter(range(1,151),distance_matrix1[:,0])
f1.show()
plt.show()

#Generate two time series data by the code: X = mvnrnd([0;0],[1 .3;.3 1],100)

#QUESTION 3.6: Plot the generated two time series in one plot

mean = [0,0]
cov = [[1, 0.3], [0.3, 1]]
x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.plot(x,'r--',y,'b--')
plt.title("X and Y Time Series")
plt.show()

#QUESTION 3.7: Calculate the T-statistics distance between the two time series. 

'''CALLING THE FUNCTION CREATED TO EVALUATE DISTANCES 
BETWEEN THE TWO TIME SERIES X AND Y'''

T_StatDist=tstat(x,y)
print(T_StatDist)

#QUESTION 3.8: Correlation of the two time series

Timeseries=np.column_stack((x,y))
Tcorr=Timeseries.corr(rowvar=False)

#QUESTION 3.9: Normalize the feature matrix of the IRIS dataset such that after normalization 
#each feature has a mean of 0 and a standard deviation of 1.  

'''Standardizing the Iris dataset so that it has normalized to 
zero mean and unit standard deviation'''

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(iris1)
NormalIris = pd.DataFrame(x_scaled)
print (NormalIris) 

                         #     END OF HOMEWORK 1     #
