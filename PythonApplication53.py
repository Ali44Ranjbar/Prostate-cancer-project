from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

df=pd.read_csv("Prostate_cancer_data1.csv")
print(df)



data=pd.DataFrame({
    "id":df.data[ : , 0],
    "radius":df.data[ : , 1],
    "texture":df.data[ : , 2],
    "perimeter":df.data[ : , 3],
    "area":df.data[ : , 4],
    "smoothness":df.data[ : , 5],
    "compactness":df.data[ : , 6],
    "symmetry":df.data[ : , 7],
    "fractal_dimension":df.data[ : , 8],
    "diagnosis_result":df.target
})

F=data[["id","radius","texture","perimeter","area","smoothness","compactness","symmetry","fractal_dimension", "diagnosis_result"]]
T=data[["diagnosis_result"]]
print(F)
print(T)

F_train,F_test,T_train,T_test=train_test_split(F,T,test_size=0.2)

from sklearn import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
model.fit(F_train,T_train)
p=model.predict(F_test)
print("-----Predict")
print(p)
print("----Real")
print(T_test)

from sklearn import metrics
print("Accuracy is :",metrics.accuracy_score( p , T_test)*100)      

print(model.predict([[2,3,4,5]]))

feature_imp = pd.Series(model.feature_importances_ , index= df.feature_names).sort_values(ascending = False)
print(feature_imp)