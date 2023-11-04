import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import streamlit as slt
plt.style.use('fivethirtyeight')

slt.title('Stock Price Prediction')


stocks = ('SBIN.NS.csv', 'TCS.csv', 'TATAMOTORS.NS.csv', 'ADANIPORTS.NS.csv','RELIANCE.NS.csv', 'M&M.NS.csv', 'HEROMOTOCO.NS.csv', 'BHARTIARTL.NS.csv')

user_input = slt.selectbox('Select Dataset For Prediction',stocks)


df = pd.read_csv(user_input)                    
df = df.dropna()
slt.subheader('Data from 2012 - 2022')
slt.write(df.head(7))
slt.subheader('Data Description', user_input)
slt.write(df.describe())


# Plotting the Closing Price
slt.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.title('Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(df.Close)
slt.pyplot(fig)

# ---------------------------------------------LSTM------------------------------------------------------------------
# Extracting the Close column
data = pd.DataFrame(df['Close'])
data_set = data.values

# Scalling using MinMax Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaleddata = scaler.fit_transform(data_set)
train_size = int(len(data_set) * 0.80)

# Training Dataset
train_data = scaleddata[:train_size, :]

# Load the Model
model = load_model('our_model.h5')

# Test Data
test_data = scaleddata[train_size-100:, :]
x_test = []
y_test = data_set[train_size:, :]
for i in range(100, len(test_data)):
    x_test.append(test_data[i-100:i, 0])


# Scalling of test data
x_test = np.array(x_test)

# Prediction
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)

# Plotting of the Predicted Values
train = data[:int(len(train_data))]
valid = data[int(len(train_data)):]

valid['Prediction'] = pred

slt.header('Prediction vs Original')
slt.subheader('LSTM')
fig2 = plt.figure(figsize=(18, 9))
plt.title('LSTM Model')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Prediction']])
plt.legend(['Train', 'Validation', 'Prediction'], loc='upper left')
slt.pyplot(fig2)


# Accuracy and Scores
slt.subheader('Accuracy and Other Scores For LSTM')
rmse = np.sqrt(np.mean(pred-y_test)**2)
slt.write('RMSE Value: ', rmse)
errors = abs(pred - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
slt.write('Accuracy: ', accuracy)

#--------------------------------------------Regression-----------------------------------------------

df_r = pd.read_csv(user_input)
df_r = df_r.drop('Adj Close',axis = 1)
df_r.dropna(inplace=True)

X=df_r.drop(labels=['Date','Close','Volume'],axis=1)
Y = df_r['Close']


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# Automating Model training Process
models = {
    'Decision Tree':DecisionTreeRegressor(),
    'Random Forest':RandomForestRegressor(),
    'Extra Tree Regression':ExtraTreesRegressor(),
}


def evaluate_model(X_train,Y_train,X_test,Y_test,models):
    
    report_a2 = {}
    report_acc = {}
    report_comp = {}
    for i in range(len(models)):
        model = list(models.values())[i]
        
        # Train model
        model.fit(X_train,Y_train)


        # Predict Testing data
        y_test_pred =model.predict(X_test)

        # Get accuracy for test data prediction

        test_model_score = round(r2_score(Y_test,y_test_pred),4)

        # Calculating the Accuracy

        errors = abs(y_test_pred - Y_test)
        mape = 100 * (errors / Y_test)
        accuracy = 100 - np.mean(mape)

        report_a2[list(models.keys())[i]] =  test_model_score
        
        report_acc[list(models.keys())[i]] =  round(accuracy,2)

        compare = pd.DataFrame({
            'Close':Y_test,
            'Predicted':y_test_pred
        })
        report_comp[list(models.keys())[i]] = compare      


    return report_a2,report_acc,report_comp

score,accuracy,compare = evaluate_model(X_train,Y_train,X_test,Y_test,models)

DT_df = pd.DataFrame(compare['Decision Tree'])
RF_df = pd.DataFrame(compare['Random Forest'])
ETR_df = pd.DataFrame(compare['Extra Tree Regression'])

DT_df = DT_df.sort_index(ascending = True)
RF_df = RF_df.sort_index(ascending = True)
ETR_df = ETR_df.sort_index(ascending = True)



slt.header('Regression Models')
slt.subheader('Random Forest')
Reg_fit1 = plt.figure(figsize=(18,8))
plt.title('Random Forest Model')
plt.plot(RF_df['Close'])
plt.plot(RF_df['Predicted'])
slt.pyplot(Reg_fit1)

slt.subheader('Result')
slt.write("R2 Score: ",score['Random Forest'])
slt.write("Accuracy",accuracy['Random Forest'])

slt.subheader('Decision Tree')
Reg_fit2 = plt.figure(figsize=(18,8))
plt.title('Decision Tree Model')
plt.plot(DT_df['Close'])
plt.plot(DT_df['Predicted'])
slt.pyplot(Reg_fit2)

slt.subheader('Result')
slt.write("R2 Score: ",score['Decision Tree'])
slt.write("Accuracy",accuracy['Decision Tree'])

slt.subheader('Extra Tree Regression')
Reg_fit3 = plt.figure(figsize=(18,8))
plt.title('Extra Tree Regression Model')
plt.plot(ETR_df['Close'])
plt.plot(ETR_df['Predicted'])
slt.pyplot(Reg_fit3)

slt.subheader('Result')
slt.write("R2 Score: ",score['Extra Tree Regression'])
slt.write("Accuracy",accuracy['Extra Tree Regression'])


#------------------------------------------------------------------------------------------------

