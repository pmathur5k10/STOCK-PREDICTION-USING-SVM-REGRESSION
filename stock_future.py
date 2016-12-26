import requests
#helps in http request
import simplejson
#used or json handling , better than json as it gives specific error code for incompatible json
import re
#used for string operations
import operator
#used for numerical operations
import sys
#used for system calls
import urllib
#used to get or push http url requests
import os
#used for instructing os to perform functions on file system
import csv
#handle csv files
import numpy as np
#used for scientific calculation
from sklearn.svm import SVR
#used for ml
import matplotlib.pyplot as matplt
#used for plotting graph


#INPUT THE COMPANY NAME
if(len(sys.argv)<2):
	print('invalid string')
	exit()

query=sys.argv[1]
#ACCESS THE YAHOO FINANACE URL FOR STOCK CODE
yahoo_stock_code="http://d.yimg.com/autoc.finance.yahoo.com/autoc?query="
yahoo_excess_code="&region=1&lang=en"
stock_url=yahoo_stock_code+query+yahoo_excess_code
response=requests.get(stock_url)

#CONEVRT THE JSON FILE INTO UTF-8 FORMAT FOR PARSING
data=simplejson.loads(response.content.decode("utf-8"))

#FETCH THE FIRST COMPANY CODE
code=data['ResultSet']['Result'][0]['symbol']
print(code)

#ACCESS THE YAHOO FINANACE API FOR DOWNLOADING THE DATA
base_url = "http://ichart.finance.yahoo.com/table.csv?s="
dataset_url=base_url+code

#CREATE PATH FOR THE DOWNLOADED FILE
output_path="C:/ml/stock_prediction/"
output_path_new=output_path+code+"_new.csv"
output_path=output_path+code+".csv"


#USE URLLIB FOR THE DATA FILE DOWNLOAD
try:
	urllib.urlretrieve(dataset_url,output_path)
except urllib.ContentTooShortError as p:
	outfile=open(output_path,"w")
	outfile.write(p.content)
	outfile.close()
	

#WRITE THE FIRST i LINES OF DATA INTO A NEW FILE 
a=open(output_path,"rb")
b=open(output_path_new,"wb")
reader=csv.reader(a,delimiter=',')
f=csv.writer(b)

i=1
for line in reader:
	if i>20:
		break
	else:
		f.writerow(line)
		i=i+1	




a.close()
b.close()
os.remove(output_path)

#COVERT THE DATA INTO TWO ARRAYS OF DATE AND PRICE

date=[]
price=[]
with open(output_path_new) as datasource:
	r=csv.reader(datasource)
	next(r)
	for row in r:
		date.append(int(row[0].split('-')[2]))
		price.append(float(row[1]))

#reshape the date array into the numpy array of nX1
date=np.reshape(date,(len(date),1))
#print(date)
#print(price)



#DATA PREDICTION
#initialise svr, fit the models, predidct the model values
#kernel specifies the kernel type used in algorithm
#C is the error penalty
#gamma is the kernel coefficient
#degree is the polynomial degree in poly kernel



svr_lin=SVR(kernel='linear',C=1e3)
svr_poly=SVR(kernel='poly',C=1e3,degree=2)
svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.1)

svr_lin.fit(date,price)
svr_poly.fit(date,price)
svr_rbf.fit(date,price)



#PLOT THE DATA ON THE GRAPH

matplt.scatter(date,price,color='black',label='data')
matplt.plot(date,svr_lin.predict(date),color='blue',label='Linear SVR')
matplt.plot(date,svr_poly.predict(date),color='red',label='Polynomial SVR')
matplt.plot(date,svr_rbf.predict(date),color='green',label='RBF SVR')
matplt.xlabel('Dates')
matplt.ylabel('Price')
matplt.title('Support Vector Regression')
matplt.legend()
matplt.show()

svr_lin.predict(10)[0]
svr_poly.predict(10)[0]
svr_rbf.predict(10)[0]



os.remove(output_path_new)
print('prediction over')








