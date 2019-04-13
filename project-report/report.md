# Gun Violence in USA
| Uma M Kugan
| umakugan@iu.edu
| Indiana University Bloomington
| hid: hid-sp18-710
| github: [:cloud:](https://github.com/cloudmesh-community/hid-sp18-710/tree/master/project-report/report.md)
| code: [:cloud:](https://github.com/cloudmesh-community/hid-sp18-710/tree/master/project-code)

---

Keywords: Gun Violence, Scikit, Fbprophet, Decision Tree, KNN, Random Forest

---

##Abstract
Gun Violence is always been the topic of debate and concern. Gun violence also affects
more than its victims. In areas where it is prevalent, just the threat of violence makes
neighborhoods poorer. Guns are also involved in suicides and accidents. Exploring 
the data set will help us find an answer for our fear what if me and people around
me are likely to be killed by gun. Even though it is an unnecessary fear, it will 
at least give us deeper understanding of how the data looks like and draw some
insights. Only way to prevent gun violence is to raise the awareness of improper
handling and storage of the weapons where crime rate is high. In this paper
gun violence data has been analyzed to understand if there are any patterns or trend in
increase in violence and also to predicting if the shooting victim is white or black.

##Introduction
Gun Violence in America According to CDC, "One person is killed by a firearms every 17 minutes, 
87 people are killed during an average day, and 609 are killed every week". 
Weapon brutality regardless of whether a man slaughters, suicides, or mischances-slaughters 
around 30,000 Americans consistently and harms 70,000 more [@gunviolence]. 
At 31.2 passing's for every million individuals, Americans are nearly as prone to pass on toward
the finish of a weapon as they are in an auto collision [@gunviolence]. 
There is adequate worldwide research that proposes the accessibility of weapons expands the 
danger of deadly brutality. At the point when weapons are available, suicide endeavors 
will probably succeed and ambushes will probably progress toward becoming crimes. 
Some exploration has shown that amassing weapons and the interest
with firearms is a pointer of standoffish conduct. 

##Relevant Existing Work 
We can find in the internet there are some organization who are collecting or 
using data from CDC to understand the trending and present data in various form
for everyone to easily understand [@national1977vital]. Mother Jones have 
investigated and analyzed data from 1982-2018 on US Mass Shootings [@follman2012guide]. 
Fivethirtyeight.com have an interactive graphics as part of their project to explore
the more than 33,000 annual gun deaths in America [@casselman2016gun]. Every-town 
gun safety is trying to get crowd  source funding and analyzing the various facts 
and issues related to gun safety and accidents [@everytown]. All these organizations, 
are trying to draw some insights from the data sets and see if they can predict and stop the
violence before it occurs.

##Data Collection and Preparation
The dataset we used in this project is collected from five thirty eight's gun deaths in America
using the R code from "https://data.world/azel/gun-deaths-in-america" [@data_world].
The dataset has data from the year 2013-2017.
Data preparation is the process of transforming raw data to draw some valuable
insights or make some predictions based on the past. Data Preparation will be
difficult if the data has improper values or nulls. 
We used Python which is a high-level object-oriented programming language, most
popular due to high availability of large collection of free libraries for this 
project.

##The Dataset
The dataset contains various information about victims of gun violence. Each row of the data set contains
the year and month of the shooting, the intent of the shooter, whether the police were at the
scene or not, the gender, age race and education level of the victim and finally the place 
where the shooting happened. The dataset also has the specific column that says if the victim is Hispanic or not.
The dataset consists of the following attributes:

	- Year: It is a numerical field and holds the year of the shooting.
	
	- Month: It is a numerical field and holds the month of the shooting.
	
	- Intent: It is a text field and specifies the intent of the shooter.
	 
	- Police: It is a numerical field and contain values 0 or 1 which indicates if the police were at the scene or not.
	
	- Sex: It is a text field and contains values M for Male and F for Females.
	 
	- Place: It is a text field and specifies where the shooting has happened
	
	- Education: It is a text field and has the education level of the victim.
	
	- Race: It is a text field and possible values are White,Black,Hispanic,Asian/Pacific Islander and Native American/Native Alaskan.
	
	- Age: It is a numeric field and specifies the age of the victim.
	
	- Hispanic: It is a numeric field and specifies if the victim is Hispanic or not. 

![image](C:\Users/gauth/Desktop/Project/images/actual_dataset.PNG)
*Figure 1 - Snapshot Of Actual Dataset*

##Technologies Used
**Pandas:** Pandas is an open source library that provides tools for data mining and
analysis using Python. It is mainly used in this project to prepare the data for
consumption by specific machine learning algorithms.

**NumPy:** NumPy is a Python library that can handle multidimensional data and
perform scientific and mathematical operations on the same. NumPy was used in this
project to perform some basic mathematical operations.

**Scikit-learn:** scikit-learn [16] is an open-source Python machine learning library
which provides numerous classification, regression and clustering algorithms. This
library was used in this project to perform the actual task of model building and
prediction. It provides a variety of evaluation metrics to validate the performance of
the model, which makes it a valuable tool.

**Prophet:** Prophet is a procedure for forecasting time series data based on an additive 
model where non-linear trends are fit with yearly, weekly, and daily seasonality, 
plus holiday effects. It works best with time series that have strong seasonal effects 
and several seasons of historical data. Prophet is robust to missing data and shifts 
in the trend, and typically handles outliers well [@taylor_prophet].

##Data Preprocessing
The original dataset is modified to create a new dataset where new columns are
added, existing columns are transformed and outliers are handled. The columns are added,
deleted or transformed based on the graphical analysis which has been performed on the data
before actually building a model.

In this project we want to if the victim is white or black. There are totally five
classes but we are predicting only two. The main reason to ignore other classes is because
from the given data set,the rest of the classes, is less than 11% of the dataset.
	
	White                             64.013604
	Black                             24.472467
	Hispanic                           9.181200
	Asian/Pacific Islander             1.388112
	Native American/Native Alaskan     0.944617

Since we are interested in only two classes, we are removing the rest of the classes from
the data before we train the model. Also there is are columns such as police and hispanic
which is not relevant to our prediction classes and hence we ignored those columns as well
from the actual data set.  

##Label Encoding
Label encoding is the technique that are used to convert categorical data, or 
text data, into numbers, so that our predictive models can better understand.
Some of the columns in our data set contains text data. In order to run any machine learning
model against the data, we can’t have text in our data. So before we run any kind of model,
we need to prepare this data by converting categorical text data into model-understandable 
numerical data, we use the Label Encoder class. The Label Encoder class from the sklearn library, 
fit and transform the data, and then replace the existing text data with the new encoded data [@medium_enocde].

![image](C:\Users/gauth/Desktop/Project/images/afterlabelencode.PNG)
*Figure 2 - After Label Encode*

##Data Slicing
Data slicing is the process to split data into train and test set. Training data set 
can be used specifically for our model building. Test data set should not be mixed up while 
building model. We can use sklearn’s train_test_split method to split the data into random 
train and test subsets of data. The three main parameters data set, test_size which represents 
what percentage from the whole data set is test data and random_state variable is a 
pseudo-random number generator state used for random sampling. After we split the data 
set into train and test data, the machine learning algorithms are applied on training data.

##Decision Trees
Decision Trees are an important type of algorithm for predictive modeling machine learning.
Decision trees are simple and can be easily trained with few hyper-parameter and easy to
interpret. The major drawback of decision tree is that they tend to over fit the data.  
Decision Tree classifiers use decision trees to make a prediction about the value
of a target variable. The decision trees are basically functions that successively
determine the class that the input needs to be assigned. A decision tree contains a 
root node, interior nodes and leaf nodes [@decision_tree1]. The interior nodes are the 
splitting nodes, i.e. based on the condition specified in the function at these nodes, the tree is 
split into two or more branches.The main advantage in a decision tree classifier is that
an input is tested against only specific subsets of the data which eliminates unnecessary
computations [@decision_tree2]. Another advantage of Decision Trees is that we can use a feature selection
algorithm to make a decision on which features can be used for the decision tree classifier. 
The lesser the number of features, better the efficiency of the algorithm [@decision_tree1]. 

The important parameters of decision tree classifiers used in our model:

	criterion: It defines the function to measure the quality of a split. Sklearn supports gini 
				criteria for Gini Index and entropy for Information Gain. The default is gini value.

	max_depth: The max_depth parameter denotes maximum depth of the tree.
	
	min_samples_leaf: The minimum number of samples required to be at a leaf node. 

The accuracy we got from implementing the decision model is 87%. Accuracy is the ratio of the 
correctly predicted data points to all the predicted data points. The value of accuracy determines
the effectiveness of our algorithm.

![image](C:\Users/gauth/Desktop/Project/images/decision_tree.PNG)
*Figure 3 - Decision Tree* 

##K- Nearest Neighbor
According to the K-Nearest Neighbor (KNN) algorithm, data is classified into
one of the many categories by taking a majority vote of its neighbors. 
It keeps all the training data to make future predictions by computing the similarity
between an input sample and each training instance. We identify the neighbors 
closest to the new point we wish to classify and based on these neighbors, 
we predict the label of this new point. For computing the distance measures 
such as Euclidean distance, Hamming distance or Manhattan distance will be used. 
Model picks K entries which is the number of neighbors to consider which are closest
to the new data point. Then it does the majority vote i.e the most common class/label 
among those K entries will be the class of the new data point [@knn_algo].

The Figure 3 illustrates a simple KNN classifier. Here, if k = 3, the green circle in
question would be categorized as a red triangle. If k=5, then it would be categorized
as a blue square.

![image](C:\Users/gauth/Desktop/Project/images/knn_classifier.PNG)
*Figure 3 - KNN Classifier* 


##Random Forest
Random forest classifier creates multiple decision trees from a randomly selected 
subset of the training set and then aggregates them to decide the final class of
the test object. We need first to choose random samples from a given data set, 
construct a decision tree for each sample and get a prediction result from each
decision tree. Then Perform a vote for each predicted outcome and prediction
with the most votes as the final prediction [@breiman2001random].

![image](C:/Users/gauth/Desktop/Project/images/random_f.png)
*Figure 4  Random Forest [@medium_ref8]*

##Prophet
Prophet is very powerful and effective in time series forecasting. There are 
few tools that are in industry for forecasting. We can compare these tools 
and use the one that gives the best predictions with the least amount of errors.
In this project we used prophet as it is easier to implement. Prophet can be 
installed using pip in Python. Prophet depends on a Python module called
pystan which will be installed automatically when prophet is installed.
When implementing the model, first we need to create an instance of the Prophet
class and then fit it to our data set.The future is predicted using the method
make_future_dataframe method by passing the attributes and frequency.
The forecast dataframe has important columns: yhat, yhat_lower and yhat_upper. 
yhat is our predicted forecast, yhat_lower is the lower bound for our predictions 
and yhat_upper is the upper bound for our predictions [@kd_prophet].
We can also measure the forecast error using the historical data by comparing the 
predicted values with the actual values. The cross_validation method allows us to
do this in Prophet.

##Experimental Results
All three models performed well, But the accuracy of single decision tree model
was higher than the other two models.

![image](C:/Users/gauth/Desktop/Project/images/model_perfm.png)
*Figure 5 Model Results*

Confusion matrix can also be used to display or describe the
performance of the model. It contains the information about actual and
predicted classification calculated by the machine learning model.

From the below two graphs which was generated from the output of Prophet - Forecasting
method, it clearly shows that there is increase in tend for gun violence in next five years.

![image](C:/Users/gauth/Desktop/Project/images/forecast_year.png)
*Figure 6 Forecast for next five years*


![image](C:/Users/gauth/Desktop/Project/images/forecast_month.png)
*Figure 7 Forecast for next five years by year and month*


##Graphical Analysis Results
In this section, we would look at the results of graphical analysis which
helped decide what features to include for predictions. After analyzing the data,
we deleted the column police as  it does not contribute to our classification.
From the bar plot below, it is obvious that majority of the data is for black or white victims.
![image](C:/Users/gauth/Desktop/Project/images/race.png)
*Figure 8 Race*

![image](C:/Users/gauth/Desktop/Project/images/year.png)
*Figure 9 Year*

![image](C:/Users/gauth/Desktop/Project/images/intent.png)
*Figure 10 Race*

![image](C:/Users/gauth/Desktop/Project/images/location.png)
*Figure 11 Location*

![image](C:/Users/gauth/Desktop/Project/images/incidentcnt.png)
*Figure 12 Incident Count*

![image](C:/Users/gauth/Desktop/Project/images/sex.png)
*Figure 13 Sex Of Victims*

From the above graphs we can infer that most of the victims are male and the number
of shootings have been slightly increasing year after another. The majority of the
gun deaths occur at home, about 10% occur on the streets and there are very few 
victims with higher education levels.

The Figure below shows that for black and white victim, the intent seems to be a
deciding factor, and the other three factors being age, gender and place. 
The education levels seems to have very less effect on deciding the race of the victim. 

![image](C:/Users/gauth/Desktop/Project/images/feature.png)
*Figure 14 Feature*


##Limitations
In this project we have implemented simple single binary classification model.
This project can be further enhanced and extended to build a multi classification
model for the entire CDC multiple causes of death dataset 

##Conclusion
In this project, a detailed analysis of gun violence was conducted and prediction 
models were trained using three machine learning algorithms and the trending forecast was 
also implemented and studied.It would be interesting to study if there are more factors
like population data, gun laws data that contributes to the violence.


##Acknowledgements
The author would like to thank Dr.Gregor von Laszewski for his continued support and 
suggestions in writing this report and successfully completing the project.
This project would not have been complete without his dedicated support and encouragements
and our sincere thanks to all our fellow students.
