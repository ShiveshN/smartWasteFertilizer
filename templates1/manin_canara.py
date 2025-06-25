from flask import Flask, render_template,request,session,flash
import sqlite3 as sql
import os
import pandas as pd
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import serial
import numpy as np
import csv
import math
import statistics
import easygui
import datetime
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/gohome')
def homepage():
    return render_template('index.html')







@app.route('/enternew')
def new_user():
   return render_template('signup.html')

@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            nm = request.form['Name']
            phonno = request.form['MobileNumber']
            email = request.form['email']
            unm = request.form['Username']
            passwd = request.form['password']
            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("INSERT INTO agriuser(name,phono,email,username,password)VALUES(?, ?, ?, ?,?)",(nm,phonno,email,unm,passwd))
                con.commit()
                msg = "Record successfully added"
        except:
            con.rollback()
            msg = "error in insert operation"

        finally:
            return render_template("result.html", msg=msg)
            con.close()

@app.route('/userlogin')
def user_login():
   return render_template("login.html")

@app.route('/logindetails',methods = ['POST', 'GET'])
def logindetails():
    if request.method=='POST':
            usrname=request.form['username']
            passwd = request.form['password']

            with sql.connect("agricultureuser.db") as con:
                cur = con.cursor()
                cur.execute("SELECT username,password FROM agriuser where username=? ",(usrname,))
                account = cur.fetchall()

                for row in account:
                    database_user = row[0]
                    database_password = row[1]
                    if database_user == usrname and database_password==passwd:
                        session['logged_in'] = True
                        return render_template('home.html')
                    else:
                        flash("Invalid user credentials")
                        return render_template('login.html')

@app.route('/predictinfo')
def predictin():
   return render_template('info.html')

@app.route('/predictinfo1')
def predictin1():
   return render_template('info2.html')


@app.route('/predict1',methods = ['POST', 'GET'])
def predcrop1():
    if request.method == 'POST':
        comment1 = request.form['comment1']
        comment2 = request.form['comment2']
        comment3 = request.form['comment3']
        comment4 = request.form['comment4']
        data1 = int(comment1)
        data2 = int(comment2)
        data3 = int(comment3)
        data4 = int(comment4)
        # type(data2)
        print(data1)
        print(data2)
        print(data3)
        print(data4)
        # Load the dataset
        df = pd.read_csv('fertilizer_data.csv')

        # Split data into features (X) and target (y)
        X = df[["Sensor1", "Sensor2", "Sensor3", "Sensor4"]]
        y = df["FertilizerConverted"].map({"Yes": 1, "No": 0})  # Encode labels as 1/0

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = rf.predict(X_test)
        print('y_pred', y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")

        # Function to classify given sensor values
        def classify_sensor_values(data1, data2, data3, data4):
            try:
                # Validate sensor values
                sensor_values = [data1, data2, data3, data4]
                if all(0 <= value <= 1023 for value in sensor_values):
                    # Predict the classification
                    prediction = rf.predict([sensor_values])[0]
                    result = "Yes" if prediction == 1 else "No"
                    print(f"\nFertilizer Converted: {result}")
                    return result
                else:
                    raise ValueError("Sensor values must be in the range 0-1023.")
            except ValueError as e:
                print(f"Invalid input: {e}")
                return None

        

        # Call the function with the sensor values
        classification_result  = classify_sensor_values(data1, data2, data3, data4)
        if classification_result:
            print(f"\nFertilizer Converted: {classification_result}")
            print(classification_result)  # Print Yes or No
            ret1 =  {classification_result}
            ret = classification_result
            return render_template('resultpred.html', prediction = ret1, prediction1 = ret1)


@app.route('/predict',methods = ['POST', 'GET'])
def predcrop():
    if request.method == 'POST':

        import serial
        import numpy as np
        import csv
        import math
        import statistics

        # import easygui
        import datetime
        import random
        import matplotlib.pyplot as plt
        # num = input("Enter a number: ")
        # Importing Libraries
        import serial
        import time
        i = 0
        test_str = 0
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # xx = 1
        # num = input("Enter a number: ") # Taking input from user
        arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1)
        num = str(1)
        while True:
            arduino.write(bytes(num, 'utf-8'))
            time.sleep(0.5)
            data = arduino.readline()
            data1 = str(data)
            hh = data1[2:][:-5]
            print('value', hh)


            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

            # Removing punctuations in string
            # Using loop + punctuation string
            #for ele in hh:
                #if ele in punc:
                    #test_str = hh.replace(ele, "")
                    #print('test_str', test_str)
            x[i] = hh
            print('x',x)
            i = i + 1

            if i == 40:
                break
        x = list(filter(None,x))
        x = [int(n) for n in x]
        print('array12', x)
        jj = []
        kk = []
        ll = []
        mm = []
        jj = [x[2], x[6], x[10], x[14], x[18]]
        kk = [x[3], x[7], x[11], x[15], x[19]]
        ll = [x[4], x[8], x[12], x[16], x[20]]
        mm = [x[5], x[9], x[13], x[17], x[21]]

        mq1 = math.floor(statistics.mean(jj))
        mq2 = math.floor(statistics.mean(kk))
        mq3 = math.floor(statistics.mean(jj))
        mq4 = math.floor(statistics.mean(kk))
        
        print('mq1', mq1)
        print('mq2', mq2)
        print('mq3', mq3)
        print('mq4', mq4)

        import pandas
        from sklearn import tree
        from sklearn.tree import DecisionTreeClassifier

        df = pandas.read_csv("fertilizer_data.csv")

        d = {'No': 0, 'Yes': 1}
        df['FertilizerConverted'] = df['FertilizerConverted'].map(d)

        features = ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']
        X = df[features]
        y = df['FertilizerConverted']

        dtree = DecisionTreeClassifier()
        dtree = dtree.fit(X, y)
        prediction12 = dtree.predict([[mq1, mq2, mq3, mq4]])

        print(dtree.predict([[mq1, mq2, mq3, mq4]]))

        print("[0] means 'No'")
        print("[1] means 'Yes'")
        

        if prediction12 == 0:
            ret = 'FertilizerConverted'
            ret1 = 'No'
            
            print('Fertilizer Not Converted')
            easygui.msgbox("No", title="Fertilizer Converted or Not: ")
        elif prediction12 == 1:
            ret = 'FertilizerConverted '
            ret1 = 'YES'
            print('Fertilizer Converted')
            easygui.msgbox("YES", title="Fertilizer Converted or Not: ")
        

        # make up some data
        x = [datetime.datetime.now() + datetime.timedelta(hours=i) for i in range(5)]
        # y = [3, 4, 5, 6, 7, 3, 4,9,2,5]
        # y = [i+random.gauss(0,1) for i,_ in enumerate(x)]
        # print(y)
        print('x',x)
        print('kk', kk)

        # plot
        plt.plot(x, kk)
        plt.xlabel('Date-Time')
        plt.ylabel('mq1')
        plt.title('Real time air quality')
        # beautify the x-labels
        plt.gcf().autofmt_xdate()

        #plt.show()

        x = [datetime.datetime.now() + datetime.timedelta(hours=i) for i in range(5)]
        # y = [3, 4, 5, 6, 7, 3, 4,9,2,5]
        # y = [i+random.gauss(0,1) for i,_ in enumerate(x)]
        # print(y)

        # plot
        plt.plot(x, jj)
        plt.xlabel('Date-Time')
        plt.ylabel('mq2')
        plt.title('Real time air quality')
        # beautify the x-labels
        plt.gcf().autofmt_xdate()

        #plt.show()

        return render_template('resultpred.html', prediction = ret1, prediction1 = ret1)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template('login.html')

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)

