# Step 1: Importing libraries 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# step 2: Preparing the data 
data={
    'Hours_Studied':[1,2,3,4,5,6,7,8,9,10],
    'Score':[12,25,32,40,50,55,65,73,80,90]
}
df = pd.DataFrame(data)

# Features and Target
X=df[['Hours_Studied']]  #Input (Dataframe)
y=df['Score']   #Output (series)

# Step 3: spliting the data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Step 4: Training the model
model = LinearRegression()
model.fit(X_train,y_train)

#Step 5: Make predictions
y_pred = model.predict(X_test)
print("Predictions:",y_pred)

#Step 6: Evaluate the model
mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

#Step 7: Visualizing the result 
#converting X into 1D array for ploting 
X_id = X['Hours_Studied'].values
y_pred_full=model.predict(X).flatten()

plt.scatter(X_id,y,color='blue',label='Actual Scores')
plt.plot(X_id,y_pred_full,color='red',label='Predicted line')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.title('Hours Studied vs Score Predicted')
plt.legend()
plt.show()
