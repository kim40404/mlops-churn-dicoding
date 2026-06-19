import requests
import time
import random
# Hit API 10x untuk men-generate metrics
for _ in range(10):
    requests.post('http://localhost:8000/predict', json={'gender':'Female','SeniorCitizen':0,'Partner':'Yes','Dependents':'No','tenure':12,'PhoneService':'Yes','MultipleLines':'No','InternetService':'Fiber optic','OnlineSecurity':'No','OnlineBackup':'Yes','DeviceProtection':'No','TechSupport':'No','StreamingTV':'No','StreamingMovies':'No','Contract':'Month-to-month','PaperlessBilling':'Yes','PaymentMethod':'Electronic check','MonthlyCharges':70.35,'TotalCharges':845.50})
    time.sleep(1)
print('Inference done!')