
import numpy as np
import time
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Vector_EN import Vector_ElasticNet as VEN

cant_l=5
posibles_lambdas=np.logspace(0, 4, cant_l) / 10
print('grilla de lambdas: ',posibles_lambdas)

'''
simulo una regresión y elijo el mejor lambda de la grilla para esta regresion
'''
seed=7
n=100 #tamaño de la muestra
mu1, sigma1 = 0, 0.1 # mean and standard deviation
x1 = np.random.normal(mu1, sigma1, n)
mu_e1, sigma_e1 = 0, 0.01
eps1= np.random.normal(mu_e1, sigma_e1, n)
x2=x1+eps1
eps2= np.random.normal(mu_e1, sigma_e1, n)
x3=x2+eps2
# x1,x2,x3 tienen alta correlacion
mu4,sigma4=1,0.5
x4 = np.random.normal(mu4, sigma4, n) # otra informacion
mue,sigmae=0,0.001
eps = np.random.normal(mue, sigmae, n)
mu5,sigma5=0.5,0.5
x5 = np.random.normal(mu5, sigma5, n) #ruido

X=list(zip(x1,x2,x3,x4))
y=[3.0+5*X[i][0]+(-7)*X[i][1]+ 8*X[i][2]+4*X[i][3]+eps[i] for i in range(n)]


'''
quiero correr SKL tomando en cuenta solo 3 variables
'''

X2=list(zip(x1,x2,x3))


'''
Separo en train y en test y estandarizo
'''

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.33, random_state=42)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


'''
voy a coorer Lasso de sklearn para ver el mejor lambda ( mejor r2)
'''


start_time=time.time()
r2_list=[]
intercept_list=[]
coef_list=[]
for l in posibles_lambdas:
    EN=Lasso(alpha=l,random_state=0)
    EN.fit(X_train,y_train)
    intercept_list.append(EN.intercept_)
    coef_list.append(EN.coef_)
    predict_skl=EN.predict(X_test)
    r2_sklearn_EN_test=r2_score(y_test,predict_skl)
    r2_list.append(r2_sklearn_EN_test)


(m,i) = max((v,i) for i,v in enumerate(r2_list))
best_skl={'mejor lambda de la grilla (mejor r2)':posibles_lambdas[i], 'mejor r2':r2_list[i], 'intercept':intercept_list[i], 'coef':coef_list[i] }

elapsed_time = time.time() - start_time
print('tiempo SKL: ',elapsed_time)

print('mejor resultado de SKL: ',best_skl)





'''
Separo en train y en test y estandarizo
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


'''
voy a coorer Lasso de sklearn para ver el mejor lambda y r2 score
'''


start_time=time.time()
r2_list=[]
intercept_list=[]
coef_list=[]
for l in posibles_lambdas:
    EN=Lasso(alpha=l,random_state=0)
    EN.fit(X_train,y_train)
    intercept_list.append(EN.intercept_)
    coef_list.append(EN.coef_)
    predict_skl=EN.predict(X_test)
    r2_sklearn_EN_test=r2_score(y_test,predict_skl)
    r2_list.append(r2_sklearn_EN_test)


(m,i) = max((v,i) for i,v in enumerate(r2_list))
best_skl={'mejor lambda de la grilla (mejor r2)':posibles_lambdas[i], 'mejor r2':r2_list[i], 'intercept':intercept_list[i], 'coef':coef_list[i] }

elapsed_time = time.time() - start_time
print('tiempo SKL: ',elapsed_time)

print('mejor resultado de SKL: ',best_skl)

'''
penalizo nada mas las q tienen alta correlacion

'''
max_iter=10000
alpha=1


start_time=time.time()
r2_list_VEN=[]
intercept_list_VEN=[]
coef_list_VEN=[]

for l in posibles_lambdas:
    Lambda=[l,l,l,0] 
    VEN_=VEN(alpha,Lambda,max_iter)
    VEN_.fit(X_train,y_train)
    intercept_list_VEN.append(VEN_.intercept)
    coef_list_VEN.append(VEN_.coef)
    predict_VEN=VEN_.predict(X_test)
    r2_VEN_test=r2_score(y_test,predict_VEN)
    r2_list_VEN.append(r2_VEN_test)
    
    
(m,i) = max((v,i) for i,v in enumerate(r2_list_VEN))
best_VEN={'mejor lambda de la grilla (mejor r2)':posibles_lambdas[i], 'mejor r2':r2_list_VEN[i], 'intercept':intercept_list_VEN[i], 'coef':coef_list_VEN[i] }

elapsed_time = time.time() - start_time
print('tiempo VEN: ',elapsed_time)

print('mejor resultado de VEN: ',best_VEN)

'''
si ahora agrego un ruido X5
'''

X=list(zip(x1,x2,x3,x4,x5))

'''
Separo en train y en test y estandarizo
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)




'''
voy a coorer Lasso de sklearn para ver el mejor lambda y r2 score
'''


start_time=time.time()
r2_list=[]
intercept_list=[]
coef_list=[]
for l in posibles_lambdas:
    EN=Lasso(alpha=l,random_state=0)
    EN.fit(X_train,y_train)
    intercept_list.append(EN.intercept_)
    coef_list.append(EN.coef_)
    predict_skl=EN.predict(X_test)
    r2_sklearn_EN_test=r2_score(y_test,predict_skl)
    r2_list.append(r2_sklearn_EN_test)


(m,i) = max((v,i) for i,v in enumerate(r2_list))
best_skl={'mejor lambda de la grilla (mejor r2)':posibles_lambdas[i], 'mejor r2':r2_list[i], 'intercept':intercept_list[i], 'coef':coef_list[i] }

elapsed_time = time.time() - start_time
print('tiempo SKL con ruido: ',elapsed_time)

print('mejor resultado de SKL con ruido: ',best_skl)

'''
penalizo x1 x2 x3 y x5. x4 no la penalizo

'''
max_iter=10000
alpha=1


start_time=time.time()
r2_list_VEN=[]
intercept_list_VEN=[]
coef_list_VEN=[]

for l in posibles_lambdas:
    Lambda=[l,l,l,0,l] 
    VEN_=VEN(alpha,Lambda,max_iter)
    VEN_.fit(X_train,y_train)
    intercept_list_VEN.append(VEN_.intercept)
    coef_list_VEN.append(VEN_.coef)
    predict_VEN=VEN_.predict(X_test)
    r2_VEN_test=r2_score(y_test,predict_VEN)
    r2_list_VEN.append(r2_VEN_test)
    
    
(m,i) = max((v,i) for i,v in enumerate(r2_list_VEN))
best_VEN={'mejor lambda de la grilla (mejor r2)':posibles_lambdas[i], 'mejor r2':r2_list_VEN[i], 'intercept':intercept_list_VEN[i], 'coef':coef_list_VEN[i] }

elapsed_time = time.time() - start_time
print('tiempo VEN con ruido sin penalizar x4: ',elapsed_time)

print('mejor resultado de VEN con ruido sin penalizar x4: ',best_VEN)

'''
penalizo x1 x2 x3  . x4 y x5 no las penalizo

'''
max_iter=10000
alpha=1


start_time=time.time()
r2_list_VEN=[]
intercept_list_VEN=[]
coef_list_VEN=[]

for l in posibles_lambdas:
    Lambda=[l,l,l,0,0] 
    VEN_=VEN(alpha,Lambda,max_iter)
    VEN_.fit(X_train,y_train)
    intercept_list_VEN.append(VEN_.intercept)
    coef_list_VEN.append(VEN_.coef)
    predict_VEN=VEN_.predict(X_test)
    r2_VEN_test=r2_score(y_test,predict_VEN)
    r2_list_VEN.append(r2_VEN_test)
    
    
(m,i) = max((v,i) for i,v in enumerate(r2_list_VEN))
best_VEN={'mejor lambda de la grilla (mejor r2)':posibles_lambdas[i], 'mejor r2':r2_list_VEN[i], 'intercept':intercept_list_VEN[i], 'coef':coef_list_VEN[i] }

elapsed_time = time.time() - start_time
print('tiempo VEN con ruido sin penalizar x4 y x5: ',elapsed_time)

print('mejor resultado de VEN con ruido sin penalizar x4 y x5: ',best_VEN)
