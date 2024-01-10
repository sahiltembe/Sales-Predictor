import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib 


data = pd.read_csv("Advertising.csv",index_col=0)

x = data[["TV","radio","newspaper"]]
y = data["sales"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

lm = LinearRegression()

model = lm.fit(x_train,y_train)

joblib.dump(model,"mymodel.joblib")

print("----prediction success-------")