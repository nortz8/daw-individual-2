# MRT Sakay: Predicting Hourly Passenger Data in the MRT-3 (Philippines)
Predicting the number of passengers in the crowded MRT-3 is an important first step in planning future improvements and in managing foot traffic in the train stations.
In this study, we used Logistic Regression (L1 and L2), Random Forest Regression, and Gradient Boosted Regression to predict the hourly passenger count in an MRT train station.
Historical passenger data for 2016 and 2017 was collected from the Department of Transportation (DoTr) official website and weather data is from wundergound.com.
The Gradient Boosted Model with learning rate of 0.2 and max depth of 20 produced the best result with an average r-squared accuracy of 0.9436 using 10-fold cross validation.

### To create the environment and run the application, start an anaconda prompt in the cloned directory and run the following:

`conda env create --file environment.yml`  
`conda activate daw-individual-2-delnorte`  
`python application.py`  


#### Then open this link in your browser: http://127.0.0.1:8082/

<br>

#### A deployed application is also available in the link below until January 31, 2020:  
#### http://individual2-dev.ap-northeast-1.elasticbeanstalk.com/
