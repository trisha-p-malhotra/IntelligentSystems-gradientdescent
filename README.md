# IntelligentSystems-gradientdescent
Implementing gradient descent from scratch for regression using Python


Problem Statement:<br>

For this question you may use any language but you must code up the gradient descent for regression rather than using any library function.<br>

One winter I noticed that the gas mileage on my new Prius C seemed much lower than it had been over the summer, and in fact seemed to be dependent on temperature. (This would seem to be because the gas engine continues to run to keep itself warm even when not required for moving the car.) So, I collected some data from my daily drive into work during late winter/early spring. The data is here. The first column contains the outdoor temperature (as told to me by the car) and the second the miles per gallon. Since I took the same route each day and there is very little traffic to cause variation in my driving, this would seem to be about as good an experimental process as I am willing to undertake.<br>

You will notice that the data has a few rows that do not look quite like the others - the ones with asterisks are from days where I stopped at the gas station about a mile from my house (the whole drive is 14 miles) and the ones with two numbers for temperature are those days in which the temperature varied significantly during the drive (yes, 11 degrees in 14 miles and 20 minutes!). It is up to you do decide what to do with these data points.<br>

1. Use gradient descent to compute the best linear model of the data. (You may want to use Matlab or the like to confirm your result.) What is the resulting (minimum) least-squared error?<br>
2. Consider making a piece-wise linear model - specifically, a model that consists of two linear functions, one for temperatures below a threshold and another for temperatures above the threshold. Use a simple linear (brute force) search to find the best threshold and thus the best overall model. What is the resulting error?<br>
3. Use gradient descent to compute the best quadratic model (the partial derivative for the w2 weight should include an x^2 term) or some other function (probably something that has a simple derivative, like a log). How does the error compare to the previous two models? Do you think your model is more likely than the linear model?
