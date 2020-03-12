EPS=0.001
def epsilon_equal(a,b):
    return abs(a-b)<=EPS

class Vector_ElasticNet():
    '''
    Esto asume que los datos estan estandarizados
    
    Voy a llamar:
    p son la cantidad de features y n la cantidad de datos que tengo
    X es una lista de tamaño n , en cada lugar tiene una lista de tamaño p
    y es una lista de  tamaño n
    Lamda_ es un vector de tamaño p (por cada feature tengo un lambda distinto)
    
    La función fit(X,y) calcula el intercept y los coeficientes con Coordinate Descendent usando:
    'Regularization Paths for Generalized Linear Models via Coordinate Descent' - Jerome Friedman, Trevor Hastie*, and Rob Tibshirani
    pero considerando un lambda distinto para cada feature
    '''
    def __init__(self,alpha,Lambda_,max_iter,coef=None,intercept=None):
        
        
        self.alpha=alpha # alpha=1 es LASSO
        self.Lambda_=Lambda_
        self.max_iter=max_iter
        
        # tengo p+1 parametros a  determinar (el intercept y los coeficientes)
        self.coef=coef
        self.intercept= intercept
        
    def soft_thresholding(self,z,gamma):
        if z>0 and gamma<z :
            return z-gamma
        elif z<0 and gamma<-z :
            return z+gamma
        else:
            return 0.0
        
    def fit(self,X,y):
        '''
        Inicializo el intercept y los coeficientes (por si no los pasa)
        '''
        if self.intercept==None:
            self.intercept=1.0
        if self.coef==None:
            self.coef=(len(self.Lambda_))*[1.0]
        
        '''
        inicializo los residuos
        '''
        p=len(self.Lambda_)
        n=len(y)
        residuos=[y[i]- (self.intercept+sum([X[i][j]*self.coef[j] for j in range(p)])) for i in range(n)]      
        
        '''optimizador: (calibra los parametros)'''
        cambio=True 
        for it in range(self.max_iter):
            if not cambio: break
            cambio=False        
            if self.intercept !=0 :
                old_intercept = self.intercept
                self.intercept=sum(residuos)/n +self.intercept
                #Mira abajo, pero la cuenta es la misma, este es el residuo viejo + old intercept - new intercept
                #residuos=[y[i]- (self.intercept+sum([X[i][j]*self.coef[j] for j in range(p)])) for i in range(n)]
                if not epsilon_equal(old_intercept,self.intercept):
                    cambio = True
                    for i in range(n): residuos[i] += old_intercept - self.intercept
            for j in range(p):
                if self.coef[j]!= 0 :
                    z=sum(X[i][j]*residuos[i] for i in range(n))/n+self.coef[j]
                    old_coef = self.coef[j]
                    self.coef[j]=self.soft_thresholding(z,self.Lambda_[j]*self.alpha)/(1+self.Lambda_[j]*(1-self.alpha))
                    if not epsilon_equal(old_coef,self.coef[j]):
                        cambio=True
                        for i in range(n): residuos[i] += X[i][j]*(old_coef - self.coef[j]) 

        return 0
    
    def predict(self,X):
        p=len(self.coef)
        n=len(X)
        if self.intercept==None :
            y=[(sum([X[i][j]*self.coef[j] for j in range(p)])) for i in range(n)]    
            
        else :
            y=[(self.intercept+sum([X[i][j]*self.coef[j] for j in range(p)])) for i in range(n)] 
        
        return y
