import random
import time 
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

class MSRS():
    def __init__(self, fun, bounds, num_layer_arg = () ,k = 100, N = 10000):
        self.fun = fun
        self.bounds = bounds
        self.num_paras = len(bounds)
        self.k = k 
        self.N = N
        self.num_layer_arg = num_layer_arg
        self.X_sample = np.array([]) 
        self.X_resample = np.array([])
        
        self.all_results = [] 
        self.best_x = []
        self.min_f = [] 
        
        self.total_time = 0 
        
    def ReSample(self, X ,k):
        n, m = X.shape
        sampled_X = np.zeros((k, m))
        sample_index0 = random.randint(0,n-1)
        sampled_X[0,:] = X[sample_index0, :]

        for i in range(1,k):
            dist = np.sum((X-sampled_X[0,:])**2,axis = 1)
            for j in range(i):
                dist2 = np.sum((X-sampled_X[j,:])**2,axis = 1)
                dist = np.min(np.vstack((dist, dist2)), axis = 0)
            sample_index = random.choices(list(range(0,n)), dist)
            sampled_X[i,:] = X[sample_index, :]
        return sampled_X
    
    def get_population(self):
        d =len(self.bounds)
        X_sample = np.zeros((self.N, d))
        for i in range(d):
            x_sample =  np.random.uniform(self.bounds[i][0], self.bounds[i][1], self.N)
            X_sample[:,i] = x_sample
        self.X_sample = X_sample
        return X_sample
    
    def find_global_min(self):
        t_start = time.time()
        X_resample = self.ReSample(self.get_population(), self.k)
        self.X_resample = X_resample
        all_results = []

        for i in range(self.k):
            res = opt.minimize(fun= self.fun, args = (self.num_layer_arg) ,x0=X_resample[i,:], bounds=self.bounds)
            self.all_results.append([X_resample[i,:], res.x, res.fun])

        self.all_results.sort(key=lambda x:x[2] , reverse=True) 
        self.best_x = [x[1] for x in self.all_results]
        self.min_f = [x[2] for x in self.all_results]
        t_end = time.time()
        self.total_time = t_end- t_start
        # self.best_x.sort(key=lambda x:x[1] , reverse=True)
    
    def plot_res(self):
        plt.plot([i+1 for i in range(len(self.all_results))],self.min_f)
        plt.xlabel("iteration")
        plt.ylabel("min f")
        plt.title("minimal value of f"+" when k = "+str(self.k)+" num of paras= "+str(self.num_layer_arg))
        plt.show()
        print(self.best_x[-20:])
        print(self.min_f[-20:])
        print("total time :"+str(round(self.total_time,2)))
    
    def save_results(self):
        best_theta_df = pd.DataFrame(self.best_x[-20:]).round(1)  
        best_theta_df.columns = ["theta_"+str(k+1) for k in range(self.num_paras)]
        best_theta_df["poisson_rate"] = self.min_f[-20:]
        best_theta_df["total_time"] = np.floor(self.total_time*100)/100
        best_theta_df.to_csv("min_"+self.fun.__name__+"_"+str(self.k)+"+num_of_x"+str(self.num_layer_arg)+".csv")