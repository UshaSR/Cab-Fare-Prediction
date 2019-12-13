#Load libraries
library(caret)      #for data splitting function - createDataPartition
library(rpart)      #to build Decision tree regression model
library(randomForest)   #to build random forest regression model 
library(DMwR)       #to calculate regression evaluation statistics
library(ggplot2)    #for visualizations of data
library(gbm)    #to build gradient boosting model

#set working directory
setwd("C:/Users/Usha/Edwisor/Project-Cab Fare Prediction")

#to check if the working directory is set right
getwd()

#Loading the training and test data set to model and predict values
df_train = read.csv("train_cab.csv", sep=",")
df_test = read.csv("test.csv",sep=",")

#################################DATA EXPLORATION#############################################
##### Data Exploration or preparation includes,
##1. Identification of variables and their datatypes 
##2. Descriptive statistics
##3. Conversion of data types into required ones
##4. Missing Value Analysis
##5. Outlier Analysis
##6. Feature Engineering

########### 1. Identification of variables and their datatypes############

#fetch first five observations from training dataset to identify target and predictor variables 
head(df_train)

#to fetch first five observations from test dataset
head(df_test)

#to get the number of entries or dimensions of the training dataset
dim(df_train)

#to get the dimensions of the test dataset
dim(df_test)

#to get the data types of variables in training dataset 
str(df_train)

#to get the data types of variables in test dataset 
str(df_test)

#to get the unique values from training dataset
unique(df_train)

################### 2. Descriptive statistics################

#to get the summary of training dataset
summary(df_train)

#to get the list of unique values of passenger_count
table(df_train$passenger_count)

#to get the unique values of fare_amount
unique(df_train$fare_amount)

###**Observations:
##1. There are rides consisted of even passengers greater than 500 and minimum as 0. These are clear outliers and points to data inconsistency.
##2. Most of the rides consists of passengers either 1 or 2.
##3. There are null values present in passenger_count.
##4. There are 468 unique values of fare amount in the training data.
##5. There are observations in pickup_latitude, dropoff_latitude greater than 400 and less than -90.
##6. There are observations in pickup_longitude, dropoff_longitude less than -180.

#to get the distribution of test dataset
summary(df_test)

###**Observations:
##1. Passenger count consists of maximum value 6 and minimum value 1 which is within the range. 
##2. Pickup and Dropoff latitudes and longitudes are not greater than 42 and -72 which is acceptable range for latitudes and longitudes. 

################ 3. Conversion of datatypes into the required ones###############

# As observed, we have to change fare_amount from factor to numeric
df_train$fare_amount = as.numeric(as.character(df_train$fare_amount))
summary(df_train$fare_amount)

##We will have to convert passenger_count into a categorical variable because passenger_count is not a continuous variable. passenger_count cannot take continous values. and also they are limited in number if its a cab. It can be done after missing and outlier analysis are done.

##################### 4. Missing value Analysis#####################

#to get the sum of null/missing values in training set
sum(is.na(df_train))

#create dataframe to analyze missing values in training dataset
missing_val = data.frame(apply(df_train,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_count"
missing_val$Missing_percentage = (missing_val$Missing_count/nrow(df_train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1,3)]
missing_val

#to get the sum of null/missing values in test set
sum(is.na(df_test))

###**Observations:
#1. Since the percentage of missing values of passenger_count and fare_amount in training dataset is less, we can impute the missing values statistically.
#2. However, there are no missing values present in test dataset.
#3. Imputation of missing values in training dataset:
#      Imputation is a method to fill in the missing values with estimated ones. Mean / Mode / Median imputation is one of the most frequently used methods. It consists of replacing the missing data for a given attribute by the mean or median (quantitative attribute) or mode (qualitative attribute) of all known values of that variable.
# Since, fare_amount and passenger_count are numerical and categorical, we can opt for Mean/Median/Mode and KNN methods for imputation. To find the apt method out of all,
#1. Remove any random value from the column missing and replace with NA.
#2. Impute the null value with methods one by one.
#3. Select the method with output nearly equal to the original value.

## 1. Passenger count

unique(df_train$passenger_count)
df_train$passenger_count=round(df_train$passenger_count)
df_train[,'passenger_count'] = factor(df_train[,'passenger_count'], labels=(1:6))
df_test[,'passenger_count'] = factor(df_test[,'passenger_count'], labels=(1:6))

#Choosing a random value from passenger_count to replace it as NA
df_train$passenger_count[99]

#Replace 1.0 with NA
df_train$passenger_count[99] = NA
df_train$passenger_count[99]
df_train$passenger_count[99] = NA

#Impute with mode
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Mode Method
getmode(df_train$passenger_count)
df_train$passenger_count[99]

##Here, after imputation, the values are:
#Actual value - 1
#After Mode Imputation - 1
#After KNN Imputation - 1
#We can't use mode method as data will be more biased towards passenger_count = 1 as it's proportion is high in dataset.

## 2. Fare amount

#Choosing a random fare_amount value to replace it as NA
df_train$fare_amount[201]

#Replace 6 with NA
df_train$fare_amount[201] = NA
df_train$fare_amount[201]

#Impute with mean
df_train$fare_amount[201] = mean(df_train$fare_amount,na.rm=T)
df_train$fare_amount[201]
df_train$fare_amount[201] = NA
#Impute with median
df_train$fare_amount[201] = median(df_train$fare_amount,na.rm=T)
df_train$fare_amount[201]
df_train$fare_amount[201] = NA

###**Observations:
##Here, after imputation, the values are:
#Actual value = 6.0
#After Mean Imputation - 15.01
#After Median Imputation - 8.5
#After KNN Imputation - 6.32

##We will go for KNN Imputations to impute both passenger_count and fare_amount missing values

#to get the observations with missing values in passenger_count
df_train[is.na(df_train$passenger_count),]

#to get the observations with missing values in fare_amount
df_train[is.na(df_train$fare_amount),]


#KNN Imputation
df_train = knnImputation(df_train, k = 181)

#to check for null values after removing missing value rows
sum(is.na(df_train))

####################### 5. Outlier Analysis#####################
##**Based on the basic understanding of the data from statistical analysis, outliers are the values that are not in the desired range. It is obvious that the outliers present in the training data are due to incorrectly entered or measured data. Hence, we can remove those values.

## 1. Pickup and Dropoff latitudes and longitudes

##**Observation:
#pickup and dropoff latitude should be in the range of -90 to +90 and pickup and dropoff longitude should be in the range of -180 to +180. Hence, rows having pickup and dropoff latitude, longitude values out of the given range should be considered as outliers.

#fetch rows having pickup_latitude less than -90
df_train[which(df_train$pickup_latitude < -90),]

#fetch rows having pickup_latitude greater than 90
df_train[which(df_train$pickup_latitude > 90),]

#Drop the row whose pickup_latitude > 90
df_train = df_train[-which(df_train$pickup_latitude > 90),]

#fetch rows having dropoff_latitude less than -90
df_train[which(df_train$dropoff_latitude < -90),]

#fetch rows having dropoff_latitude greater than 90
df_train[which(df_train$dropoff_latitude > 90),]

#fetch rows having pickup_longitude less than -180
df_train[which(df_train$pickup_longitude < -180),]

#fetch rows having pickup_longitude greater than 180
df_train[which(df_train$pickup_longitude > 180),]

#fetch rows having dropoff_longitude less than -180
df_train[which(df_train$dropoff_longitude < -180),]

#fetch rows having dropoff_longitude greater than 180
df_train[which(df_train$dropoff_longitude > 180),]

###**Observation:
#Thus, pickup, dropoff latitude and longitude variables are within the range in training dataset and there are no outliers in pickup, dropoff latitude and longitude variables in test dataset.

## 2. Passenger count

###**Observations:
#1.Maximum number of passengers that can travel in a car can be only 6. Hence, values greater than 6 are outliers.
#2.Passenger count cannot be 0. Hence, any value less than or equal to zero is an outlier.

#get the rows whose passenger_count values are greater than 6
df_train[which(df_train$passenger_count > 6),]

#remove rows whose passenger_count is greater than 6
df_train = df_train[-which(df_train$passenger_count > 6),]

#get the rows whose passenger_count values are less than 0
df_train[which(df_train$passenger_count < 0),]

#get the rows whose passenger_count values are 0
df_train[which(df_train$passenger_count == 0),]

#remove rows whose passenger_count is equal to 0
df_train = df_train[-which(df_train$passenger_count == 0),]

#arrange passenger_count in ascending order to get outliers from passenger_count 
print(df_train[order(df_train$passenger_count,decreasing = FALSE),])


##Now, maximum and minimum values of passenger_count are within the range.

## 3.Fare amount

summary(df_train$fare_amount)

##**Observation:
#Fare amount can't be zero or less than that. Hence, values less than or equal to 0 are outliers.

#to get rows having fare_amount less than 0
df_train[which(df_train$fare_amount < 0),]

#remove rows with fare_amount value less than 0
df_train = df_train[-which(df_train$fare_amount < 0),]

#to get rows having fare_amount equal to 0
df_train[which(df_train$fare_amount == 0),]

#remove rows with fare_amount value equal to 0
df_train = df_train[-which(df_train$fare_amount == 0),]

#arrange fare_amount in descending order to get outliers from fare_amount
print(df_train[order(df_train$fare_amount,decreasing = TRUE),])

##**Observation:
#Now, there is a huge difference in the fare amounts of first two observations and the rest of the dataset. We can remove them as they are outliers.

#remove rows with fare_amount value greater than 454
df_train = df_train[-which(df_train$fare_amount >454),]

#fetch the minimum value of fare_amount
min(df_train$fare_amount)

#remove fare_amount minimum value 0.01 as value can never be this low
df_train = df_train[-which(df_train$fare_amount == 0.01),]

###**Observations:
#1. Now, maximum and minimum values of fare amount are in the range.
#2. Hence, there are no outliers present in fare_amount in training dataset. There are no missing values and outliers present in test dataset.

####################### 5. Feature Engineering####################

############Feature Creation################

## 1.Distance

##We have latitude and longitude coordinates. Thus we can calculate distance between the coordinates.This can be done based on haversine formula. The haversine formula determines the great-circle distance between two points on a sphere given their longitudes and latitudes. Here, we create a new derived feature - distance. 

deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}

#apply haversine function to training dataset
df_train$distance = haversine(df_train$pickup_longitude,df_train$pickup_latitude,df_train$dropoff_longitude,df_train$dropoff_latitude)
head(df_train)

#apply haversine function to test dataset
df_test$distance = haversine(df_test$pickup_longitude,df_test$pickup_latitude,df_test$dropoff_longitude,df_test$dropoff_latitude)
head(df_test)

#to get the distribution of distance in training dataset
summary(df_train$distance)

#to get the distribution of distance in test dataset
summary(df_test$distance)

#arrange distance values in descending order to find whether outliers are present or not
print(df_train[order(df_train$distance,decreasing = TRUE),])

###**Observations:

##We get to know that there is a drop in the value of distance from 4447.08 to 129.95 and there is no increase after it. Hence,
#1. Distance values greater than 130 could be considered as outliers and removed.
#2. Distance cannot be zero. The rows containing distance equal to 0 can also be dropped.

#drop rows whose distance value is equal to zero
df_train = df_train[-which(df_train$distance == 0),]

#drop rows whose distance value is greater than 130 (rounding off 129.95 max value)
df_train = df_train[-which(df_train$distance > 130),]

#arrange distance values in descending order to ensure there are no outliers
print(df_train[order(df_train$distance,decreasing = TRUE),])

#arrange distance values in descending order to check for oultiers in test dataset
print(df_test[order(df_test$distance,decreasing = TRUE),])

#to get the count of distance value equal to 0 in test dataset
df_test[which(df_test$distance == 0),]

#drop rows whose distance value is equal to 0
df_test = df_test[-which(df_test$distance == 0),]
print(df_test[order(df_test$distance,decreasing = TRUE),])

###**Observation:
#Thus, there are no outliers in distance variable in both training and test datasets.

## 2. Year, Month, Day, Date, Hour

## Now, we have pickup_datetime feature with us. We can generate new variables like year, month, date, day, hour from it which might have better relationship with our target variable. This also helps to highlight the hidden relationship in a variable.

df_train$pickup_date = as.Date(as.character(df_train$pickup_datetime))
df_train$pickup_day = as.factor(format(df_train$pickup_date,"%u"))# Monday = 1
df_train$pickup_month = as.factor(format(df_train$pickup_date,"%m"))
df_train$pickup_year = as.factor(format(df_train$pickup_date,"%Y"))
pickup_time = strptime(df_train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_train$pickup_hour = as.factor(format(pickup_time,"%H"))
df_train$passenger_count = as.factor(df_train$passenger_count)
summary(df_train)

#to get the observation which contains 'pickup_date' as <NA>
df_train[is.na(df_train['pickup_date']),]

####We have one <NA> value induced in pickup_date and pickup_datetime value is 43. We can remove the observation as it is an outlier and may be incorrectly entered data.

#Drop the observation that has value of 'pickup_date' as <NA>
df_train = df_train[-which(is.na(df_train['pickup_date'])),]


## Deriving new variables from pickup_datetime in test dataset.

df_test$pickup_date = as.Date(as.character(df_test$pickup_datetime))
df_test$pickup_day = as.factor(format(df_test$pickup_date,"%u"))# Monday = 1
df_test$pickup_month = as.factor(format(df_test$pickup_date,"%m"))
df_test$pickup_year = as.factor(format(df_test$pickup_date,"%Y"))
pickup_time = strptime(df_test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
df_test$pickup_hour = as.factor(format(pickup_time,"%H"))
df_test$passenger_count = as.factor(df_test$passenger_count)
summary(df_test)

############# Feature Selection ###################### 

##Correlation Matrix:

#We can find the correlation between fare_amount (target continuous variable) and independent continuous variables in training dataset using correlation matrix.
#And if features are correlated with each other that could introduce bias into our models. Hence there should be no correlation between independent features and there should be high correlation between target and independent variables.

numeric_index = sapply(df_train,is.numeric) #selecting only numeric
numeric_data = df_train[,numeric_index]

#to find the correlation between variables in training dataset
corrgram::corrgram(df_train[,numeric_index],upper.panel=corrgram::panel.pie, main = "Correlation Plot")

###**Observations:

#1. Now we have derived different variables like year, month etc from pickup_datetime variable and also we have calculated distance using pickup,dropoff latitude and longitude coordinates. 
#2. Also, from correlation matrix we can see that distance is highly correlated with fare_amount(target variable) and pickup and dropoff latitudes and longitudes variables(dependent variables) are highly negatively correlated with each other. Hence, we can drop pickup_datetime, pickup, dropoff latitude and longitude variables.

##ANOVA test:

#We an find the correlation between fare_amount (target continuous variable) and independent categorical variables present in training dataset using ANOVA test of independence. It is compares the mean between each groups in a categorical variable.

#Hypothesis of ANOVA testing :

#Null Hypothesis: Mean of all categories in a variable are same and fare_amount doesn't depend on it.
#Alternate Hypothesis: Mean of at least one category in a variable is different and fare_amount depends on it.
#If p-value is less than 0.05 then we reject the null hypothesis.
#And if p-value is greater than 0.05 then we accept the null hypothesis.


aov_results = aov(fare_amount ~ passenger_count + pickup_hour + pickup_month + pickup_year+ pickup_date + pickup_day,data = df_train)
summary(aov_results)

#We can reject null hypothesis as fare_amount is dependent on the pickup_year,pickup_month and pickup_hour.

##Dropping above mentioned variables from training dataset after analyzing all the variables
df_train = subset(df_train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
df_test = subset(df_test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
df_train = subset(df_train,select=-c(pickup_date,pickup_datetime,pickup_day,passenger_count))
df_test = subset(df_test,select=-c(pickup_date,pickup_datetime,pickup_day,passenger_count))

############# Feature Scaling #########################

##As the continuous variables, distance and fare_amount are varying in units and range,we can scale them before applying machine learning algorithms on them. We can perform normality check on variables to determine which method to be used to scale.

#plot histogram to check for distribution of distance variable in training dataset
ggplot2::ggplot(df_train, ggplot2::aes_string(x = df_train$distance)) + 
  ggplot2::geom_histogram(fill="blue", colour = "black") + ggplot2::geom_density() +
  ggplot2::theme_bw() + ggplot2::xlab("distance") + ggplot2::ylab("Frequency")+ggplot2::ggtitle(" distribution of distance")

## As the distribution of distance variable is right skewed, we can use log transformation to change the distribution.

# Lets define function for log transformation of variables
signedlog10 = function(x) {
  ifelse(abs(x) <= 1, 0, sign(x)*log10(abs(x)))
}

# Applying log function to distance variable
df_train$distance = signedlog10(df_train$distance)

#check distribution after applying log transform method
ggplot2::ggplot(df_train, ggplot2::aes_string(x = df_train$distance)) + 
  ggplot2::geom_histogram(fill="blue", colour = "black") + ggplot2::geom_density() +
  ggplot2::theme_bw() + ggplot2::xlab("distance") + ggplot2::ylab("Frequency")+ggplot2::ggtitle(" distribution of distance")

#plot histogram to check for distribution of fare_amount variable in training dataset
ggplot2::ggplot(df_train, ggplot2::aes_string(x = df_train$fare_amount)) + 
  ggplot2::geom_histogram(fill="blue", colour = "black") + ggplot2::geom_density() +
  ggplot2::theme_bw() + ggplot2::xlab("fare amount") + ggplot2::ylab("Frequency")+ggplot2::ggtitle(" distribution of distance")

## As the distribution of fare amount variable is right skewed, we can use log transformation to change the distribution.

# Lets define function for log transformation of variables
signedlog10 = function(x) {
  ifelse(abs(x) <= 1, 0, sign(x)*log10(abs(x)))
}

# Applying log function to distance variable
df_train$distance = signedlog10(df_train$fare_amount)

#check distribution after applying log transform method
ggplot2::ggplot(df_train, ggplot2::aes_string(x = df_train$fare_amount)) + 
  ggplot2::geom_histogram(fill="blue", colour = "black") + ggplot2::geom_density() +
  ggplot2::theme_bw() + ggplot2::xlab("fare amount") + ggplot2::ylab("Frequency")+ggplot2::ggtitle(" distribution of distance")

#################### Applying Machine Learning algorithms ###########################

## We can split training data into train and test datasets. Training dataset is used for building training model and test dataset is for validating our model. This is done to understand the robustness, accuracy and performance of the model built.

#Obtain test and training dataset
train.index = caret::createDataPartition(df_train$fare_amount, p = .80, list = FALSE)
train_data = df_train[train.index,]
test_data  = df_train[-train.index,]

################## 1. Linear regression ##################

lm_model = lm(fare_amount~.,train_data)
summary(lm_model)
str(test_data)
test_data[,2:5]

predictions_LM = predict(lm_model, test_data[,2:5])

ggplot2::qplot(x = test_data[,1], y = predictions_LM, data = test_data, color = I("blue"), geom = "point")

RMSE = function(act, pred){
  sqrt(mean((act - pred)^2))
}

RMSE(test_data[,1],predictions_LM)

################## 2. Decision Tree ##################

Dt_model = rpart(fare_amount~ ., data = train_data, method = "anova")

summary(Dt_model)

predictions_DT = predict(Dt_model, test_data[,2:5])

RMSE = function(act, pred){
  sqrt(mean((act - pred)^2))
}

RMSE(test_data[,1],predictions_DT)

################## 3. Random Forest ###################

rf_model = randomForest(fare_amount ~.,data=train_data, importance = TRUE, ntree = 200)

summary(rf_model)

predictions_RF = predict(rf_model, test_data[,2:5])

RMSE = function(act, pred){
  sqrt(mean((act - pred)^2))
}

regr.eval(test_data[,1],predictions_RF)

################## 4. Gradient Boosting ###################
gbm_model = gbm(fare_amount~.,data = train_data, n.trees = 200, shrinkage = 0.01,interaction.depth = 5)
summary(gbm_model)

predictions_GBM = predict(gbm_model, test_data[,2:5],n.trees = 10000)

regr.eval(test_data[,1],predictions_GBM)

########PREDICTION OF FARE AMOUNT FOR TEST DATASET##################################

##Thus,we can build prediction model for test dataset using Gradient boosting method.

gbm_model = gbm(fare_amount~.,data = train_data , n.trees = 10000, shrinkage = 0.01,interaction.depth = 5)
summary(gbm_model)

predictions_GBM = predict(gbm_model, df_test ,n.trees = 10000)

df_test$Predicted_fare_amount = predictions_GBM
