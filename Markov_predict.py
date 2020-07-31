import numpy as np
import math
import pylab as pl
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import xlrd
import xlwt


class read_Excel:
    def __init__(self,data_path,sheet_index,Sheet):
       workbook = xlrd.open_workbook(data_path)  
       sheet_name = workbook.sheet_names()[sheet_index]
       self.sheet = workbook.sheet_by_name(Sheet)   

    def read_data(self,row_index,col_index):
       dataset = []
       for j in row_index: 
         data = []
         for i in col_index:
            data.append(float((self.sheet).cell(j,i).value))
         dataset.append(data)
       return dataset 

class markov:
    def __init__(self,data,select):
       self.data = data
       self.select = select
       

    def get_data(self):
       X0 = [];Y0 = [];X1 = [];Y1 = []
       for i in range(12):
          x = [];y = []
          for j in range(len(self.data)//12):
            if (i == 11) and (j == len(self.data)//12 - 1):
                 break
            else:
              x.append(self.data[12*j+i])
              y.append(self.data[12*j+i+1])
          X0.append(x)
          Y0.append(y)
       for i in range(12):
         x = [];y = []
         for j in range(len(self.data)//12-1):
           x.append(self.data[12*j+i])
           y.append(self.data[12*(j+1)+i])
         X1.append(x)
         Y1.append(y)
       return X0,Y0,X1,Y1

    def convert_loss(self,error,data):
        ERROR = self.mat_to_list(error)
        DATA = self.mat_to_list(data)
        E = []
        for i in range(len(ERROR)):
          E.append(ERROR[i]/DATA[i])
        return np.mat(E)
   
    def mat_to_list(self,mat_x):
      list_x = []
      m,n = np.shape(mat_x)
      for i in range(m):
        for j in range(n):
          list_x.append(mat_x[i,j])
      return list_x

    def Least_Square(self,X,Y):
        weights = [];errors = [];fittings = []
        for i in range(len(X)):
          X_expand = []
          for j in range(len(X[i])):
            X_expand.append([X[i][j],1])
          X_mat = np.mat(X_expand)
          Y_mat = np.mat(Y[i])
          W = (X_mat.T * X_mat).I * (X_mat.T) * (Y_mat.T)
          error = Y_mat.T - X_mat * W
          errors.append(self.mat_to_list(error))
          weights.append(self.mat_to_list(W))
          fittings.append(self.mat_to_list(X_mat * W))
        return weights,errors,fittings

    def get_W(self,E0,E1,F0,F1):
       F0_old = [F0[-1]]+F0[0:len(F0)-1]
       E0_old = [E0[-1]]+E0[0:len(E0)-1]
       E00 = [];F00 = []
       for i in range(len(E0_old)):
         if len(E0_old[i])<len(E0_old[-2]):
            E00.append([float('inf')]+E0_old[i])
            F00.append([float('inf')]+F0_old[i])
         else:
            E00.append(E0_old[i])
            F00.append(F0_old[i])
       E00 = np.mat(E00).T
       F00 = np.mat(F00).T
       E01 = np.mat([[float('inf')]+x for x in E1]).T
       F01 = np.mat([[float('inf') for x in range(len(F1[0]))]] + F1)
       m,n = np.shape(E00)
       weights = []
       for i in range(n):
         weight = []
         for j in range(m):
           if E00[j,i] != float('inf') and E01[j,i] != float('inf'):
             if E00[j,i] * E01[j,i] > 0:
               if abs(E00[j,i]) > abs(E01[j,i]):
                 weight.append(np.array([0,1]))
               elif abs(E00[j,i]) < abs(E01[j,i]):
                 weight.append(np.array([1,0]))
               else:
                 weight.append(np.array([0.5,0.5]))
             else:
               weight.append(np.array([abs(E01[j,i])/(abs(E00[j,i])+abs(E01[j,i])),abs(E00[j,i])/(abs(E00[j,i])+abs(E01[j,i]))]))
         weights.append(sum(weight)/len(weight))
       E = []
       for i in range(m):
         e = []
         for j in range(n):
            if j in [0,1,2]:
              if E00[i,j] == float('inf') or E01[i,j] == float('inf'):
                 e.append(float('inf'))
              else:
                 #e.append(E00[i,j]*weights[j][0]+E01[i,j]*weights[j][1])
                 e.append((E00[i,j]*weights[j][0]+E01[i,j]*weights[j][1])/(F00[i,j]*weights[j][0]+F01[i,j]*weights[j][1]))
         E.append(e)
       return weights,E

    def get_initial_w1(self,w00_mat,w01_mat,w02_mat,w03_mat):
        x3 = [];x2 = [];x1 = [];y3 = [];y2 = [];y1 = []
        for k in range(len(self.data)//12-1):
          if k == 0:
             x3.append((self.data[3] - w03_mat[0,1])/ w03_mat[0,0])
             x2.append((x3[-1] - w02_mat[0,1])/ w02_mat[0,0])
             x1.append((x2[-1] - w01_mat[0,1])/ w01_mat[0,0])
          else:
             x3.append(((self.data[12*k+3] - w03_mat[0,1])/ w03_mat[0,0]+(self.expand_y(self.expand_y(self.expand_y(self.data[12*k-1])*w00_mat.T)*w01_mat.T)*w02_mat.T)[0,0])/2)
             x2.append(((x3[-1] - w02_mat[0,1])/ w02_mat[0,0]+ (self.expand_y(self.expand_y(self.data[12*k-1])*w00_mat.T)*w01_mat.T)[0,0])/2)     
             x1.append(((x2[-1] - w01_mat[0,1])/ w01_mat[0,0]+(self.expand_y(self.data[12*k-1])*w00_mat.T)[0,0])/2)
          y3.append(((self.data[12*(k+1)+3] - w03_mat[0,1])/ w03_mat[0,0]+(self.expand_y(self.expand_y(self.expand_y(self.data[12*(k+1)-1])*w00_mat.T)*w01_mat.T)*w02_mat.T)[0,0])/2)
          y2.append(((y3[-1] - w02_mat[0,1])/ w02_mat[0,0]+(self.expand_y(self.expand_y(self.data[12*(k+1)-1])*w00_mat.T)*w01_mat.T)[0,0])/2)      
          y1.append(((y2[-1] - w01_mat[0,1])/ w01_mat[0,0]+(self.expand_y(self.data[12*(k+1)-1])*w00_mat.T)[0,0])/2)
        w1,_,_ = self.Least_Square([x1,x2,x3],[y1,y2,y3])
        return np.mat(x1),np.mat(x2),np.mat(x3),w1

    def get_E(self,E0,E1,F0,F1):
       W,_ = self.get_W(E0,E1,F0,F1)
       E0 = [E0[-1]]+E0[0:len(E0)-1]
       E00 = []
       for i in range(len(E0)):
         if len(E0[i])<len(E0[-2]):
            E00.append([float('inf')]+E0[i])
         else:
            E00.append(E0[i])
       E00 = np.mat(E00).T;E01 = np.mat([[float('inf')]+x for x in E1]).T
       E = []
       m,n = np.shape(E00)
       for i in range(n):
         e = []
         for j in range(m):
           if E00[j,i] != float('inf') and E01[j,i] != float('inf'):
             e.append((E00[j,i])*W[i][0] + (E01[j,i])*W[i][1])
           else:
             if abs(E00[j,i]) > abs(E01[j,i]):
               e.append(E01[j,i])
             else:
               e.append(E00[j,i])
         E.append(e)
       
       E = np.mat(E).T
       E0_new = [];E1_new = []
       for i in range(n):
         e0_new = [];e1_new = []
         for j in range(m):
          if E00[j,i] != float('inf'):
           e0_new.append(E[j,i])
          if E01[j,i] != float('inf'):
           e1_new.append(E01[j,i])
         E0_new.append(e0_new)
         E1_new.append(e1_new)
       E0_new = E0_new[1:len(E0_new)]+[E0_new[0]]
       error = 0.0
       for x in self.mat_to_list(E):
         if x != float('inf'):
           error += abs(x)
       print('L',error)
       return E0_new,E1_new

    def BP(self,X,Y,W,E,F):
      weights = []
      errors = []
      fittings = []
      step = math.pow(10,-6) * np.mat([[math.pow(10,-10),0.0],[0.0,1.0]])
      for i in range(len(X)):
        x = np.mat(X[i]);y = np.mat(Y[i]);e = E[i];w = np.mat(W[i]);f = F[i]
        if (self.select == True) and (i in [0,1,2,11]):
           weights.append(self.mat_to_list(w))
           errors.append(e)
           fittings.append(f)
        else:
         LOSS = float('inf')
         while(1):
           predict_y = self.expand_y(x) * w.T
           delt_y = y - predict_y.T
           delt = self.convert_loss(delt_y,y)
           w += x * self.expand_y(delt) * step
           loss = np.mean(np.abs(self.mat_to_list(delt)))
           if loss < LOSS:
             LOSS = loss.copy()
             error = delt_y.copy()
             fitting = predict_y.copy()
           else:
             break
         weights.append(self.mat_to_list(w))
         errors.append(self.mat_to_list(error))
         fittings.append(self.mat_to_list(fitting))
      return weights,errors,fittings

    def expand_y(self,y_mat):
        if np.shape(y_mat) == ():
          y_list = [y_mat]
        else:
          y_list = self.mat_to_list(y_mat)
        y_new = []
        for i in range(len(y_list)):
          y_new.append([y_list[i],1])
        return np.mat(y_new)

    def adjust_W(self,W0,W1,X0,Y0,E0,E1,F0,F1):
        x = [];y = [];y_true = []
        for i in range((len(self.data)//12)-1):
          x.append([self.data[12*i+11],1])
          y.append(self.data[12*(i+1)+3])
        X = np.mat(x)
        Y = np.mat(y)
        W = (X.T * X).I * (X.T) * Y.T
        k = math.pow(W[0,0],0.25)
        b = W[1,0] / (k*k*k+k*k+k+1)
        w = [k,b]

        step = math.pow(10,1) * np.mat([[math.pow(10,-10),0.0],[0.0,1.0]])
        w00_mat = np.mat(w); w01_mat = np.mat(w); w02_mat = np.mat(w); w03_mat = np.mat(w)
        y0 = [];y1 = [];y2 = [];y3 = [];y4 = []

        L = float('inf')
        for i in range((len(self.data)//12)-1):
           y0.append(self.data[12*i+11])
           y1.append(self.data[12*(i+1)])
           y2.append(self.data[12*(i+1)+1])
           y3.append(self.data[12*(i+1)+2])
           y4.append(self.data[12*(i+1)+3])
        x1,x2,x3,w1 = self.get_initial_w1(w00_mat,w01_mat,w02_mat,w03_mat)
        w11_mat = np.mat(w1[0]); w12_mat = np.mat(w1[1]); w13_mat = np.mat(w1[2])
       
        y0 = np.mat(y0);y1 = np.mat(y1);y2 = np.mat(y2);y3 = np.mat(y3);y4 = np.mat(y4)
        n = 0;w1 = np.array([0.5,0.5]);w2 = np.array([0.5,0.5]);w3 = np.array([0.5,0.5])
        while(1):
          loss = 0.0
          if n != 0 and self.select == True:
            w1 += np.array(self.mat_to_list(np.mat([self.mat_to_list(self.expand_y(y0) * w00_mat.T),self.mat_to_list(self.expand_y(x1) * w11_mat.T)]) * delt1.T * math.pow(10,-6)))
            w1 /= sum(w1)
          y1_mat = w1[0] * self.expand_y(y0) * w00_mat.T + w1[1] * self.expand_y(x1) * w11_mat.T
          if n != 0 and self.select == True:
            w2 += np.array(self.mat_to_list(np.mat([self.mat_to_list(self.expand_y(y1) * w01_mat.T),self.mat_to_list(self.expand_y(x2) * w12_mat.T)]) * delt2.T * math.pow(10,-6)))
            w2 /= sum(w2)
          y2_mat = w2[0] * self.expand_y(y1_mat) * w01_mat.T + w2[1] * self.expand_y(x2) * w12_mat.T
          if n != 0 and self.select == True:
            w3 += np.array(self.mat_to_list(np.mat([self.mat_to_list(self.expand_y(y2) * w02_mat.T),self.mat_to_list(self.expand_y(x3) * w13_mat.T)]) * delt3.T * math.pow(10,-6)))
            w3 /= sum(w3)
          y3_mat = w3[0] * self.expand_y(y2_mat) * w02_mat.T + w3[1] * self.expand_y(x3) * w13_mat.T
         # print('w',(w1,w2,w3))
          y4_mat =self.expand_y(y3_mat) * w03_mat.T
          e1 = y1.T - y1_mat;e2 = y2.T - y2_mat;e3 = y3.T - y3_mat;e4 = y4.T - y4_mat
          e00 = y1.T - self.expand_y(y0) * w00_mat.T;e01 = y2.T - self.expand_y(y1_mat) * w01_mat.T;e02 = y3.T - self.expand_y(y2_mat) * w02_mat.T
          e10 = y1.T - self.expand_y(x1) * w11_mat.T;e11 = y2.T - self.expand_y(x2) * w12_mat.T;e12 = y3.T - self.expand_y(x3) * w13_mat.T
          ee1 = self.mat_to_list(e1);ee2 = self.mat_to_list(e2);ee3 = self.mat_to_list(e3);ee4 = self.mat_to_list(e4)
          y01 = self.mat_to_list(y1_mat);y02 = self.mat_to_list(y2_mat);y03 = self.mat_to_list(y3_mat);y04 = self.mat_to_list(y4_mat)
          ee = [[float('inf'),float('inf'),float('inf')]];yy = [[float('inf'),float('inf'),float('inf')]]
         
          for i in range(len(ee1)):
             ee.append([ee1[i]/y01[i],ee2[i]/y02[i],ee3[i]/y03[i]])
             yy.append([y01[i],y02[i],y03[i]])
          
          if self.select == True:
            UPDATE = update(self.data,ee)
            y1_new,y2_new,y3_new = UPDATE.update_y(yy)
            delt4 = self.convert_loss(e4,y4)
            delt3 = self.convert_loss(y3 - y3_new,y3)
            delt2 = self.convert_loss(y2 - y2_new,y2)
            delt1 = self.convert_loss(y1 - y1_new,y1)
          else:
            delt4 = self.convert_loss(e4,y4)
            delt3 = self.convert_loss(e3,y3)
            delt2 = self.convert_loss(e2,y2)
            delt1 = self.convert_loss(e1,y1)
          loss += np.mean(np.abs(self.mat_to_list(delt4)))
          dw03 = delt4 * self.expand_y(y3_mat)
          w03_mat += dw03 * step
          loss += np.mean(np.abs(self.mat_to_list(delt3)))
          dw02 = delt3 * self.expand_y(y2_mat)
          w02_mat += dw02 * step
          loss += np.mean(np.abs(self.mat_to_list(delt2)))
          dw01 = delt2 * self.expand_y(y1_mat)
          w01_mat += dw01 * step
          loss += np.mean(np.abs(self.mat_to_list(delt1)))
          dw00 = delt1 * self.expand_y(y0)
          w00_mat += dw00 * step

          x1,x2,x3,_ = self.get_initial_w1(w00_mat,w01_mat,w02_mat,w03_mat)
          dw13 = delt3 * self.expand_y(x3)
          w13_mat += dw13 * step
          dw12 = delt2 * self.expand_y(x2)
          w12_mat += dw12 * step
          dw11 = delt1 * self.expand_y(x1)
          w11_mat += dw11 * step

          n += 1
          print('loss',loss/4)
          if loss < L:
             L = loss.copy()
             weight00 = w00_mat.copy()
             weight01 = w01_mat.copy()
             weight02 = w02_mat.copy()
             weight03 = w03_mat.copy()
             weight11 = w11_mat.copy()
             weight12 = w12_mat.copy()
             weight13 = w13_mat.copy()
             error00 = e00.copy();error01 = e01.copy();error02 = e02.copy();error03 = ee4.copy()
             error10 = e10.copy();error11 = e11.copy();error12 = e12.copy()
             yy01 = y01.copy();yy02 = y02.copy();yy03 = y03.copy();yy04 = y04.copy()
             ww1 = w1.copy();ww2 = w2.copy();ww3 = w3.copy()
          else:
             break
          
        W0[11] = self.mat_to_list(weight00)
        W0[0] = self.mat_to_list(weight01)
        W0[1] = self.mat_to_list(weight02)
        W0[2] = self.mat_to_list(weight03)
        W1[0] = self.mat_to_list(weight11)
        W1[1] = self.mat_to_list(weight12)
        W1[2] = self.mat_to_list(weight13)
        E0[-1] = self.mat_to_list(error00);E0[0] = self.mat_to_list(error01);E0[1] = self.mat_to_list(error02);E0[2] = error03
        E1[0] = self.mat_to_list(error10);E1[1] = self.mat_to_list(error11);E0[2] = self.mat_to_list(error12)
        F0[-1] = yy01;F0[0] = yy02;F0[1] = yy03;F0[2] = yy04
        F1[-1] = yy01;F1[0] = yy02;F1[1] = yy03;F1[2] = yy04
        self.initial_W = [ww1,ww2,ww3]
        return W0,W1,E0,E1,F0,F1
    
    def get_params(self):
        X0,Y0,X1,Y1 = self.get_data()
        W0,E0,F0 = self.Least_Square(X0,Y0)
        W1,E1,F1 = self.Least_Square(X1,Y1)

        W0,W1,E0,E1,F0,F1 = self.adjust_W(W0,W1,X0,Y0,E0,E1,F0,F1)

        E0,E1 = self.get_E(E0,E1,F0,F1)
        W0,E0,F0 = self.BP(X0,Y0,W0,E0,F0)
        W1,E1,F1 = self.BP(X1,Y1,W1,E1,F1)
        
        W,E = self.get_W(E0,E1,F0,F1)
        return W0,W1,W,E

class update:
   def __init__(self,data,E):
       self.data = data
       self.E = E
       self.holiday = [35,54,43,32,51,40,29]
       

   def list_include(self,a,b):
     j = 0
     for data in a:
       if data in b:
         j+=1
     if j == len(a):
       o = True
     else:
        o = False
     return o
   
   def get_y(self,y_list):
     group = [[y_list[0]]]
     while(1):
      group_new = []
      for i in range(len(group)):
        idx = y_list.index(group[i][-1])
        o = 0
        for j in range(idx,len(y_list)):
          y_new = (group[i]).copy()
          if y_list[j] < y_new[-1]:
            y_new.append(y_list[j])
            group_new.append(y_new)
            o = 1
        if o == 0:
         group_new.append(y_new)
      if group == group_new:
        break   
      group = group_new.copy() 
     group_new = []
     for data in group:
       o = 0
       for dataset in group:
        if (self.list_include(data,dataset)==True) and (data != dataset):
          o = 1
       if o == 0:
         group_new.append(data)
     return group_new

   def get_x(self,x_list,y_list,y):
      x_group = []
      for i in range(len(y)):
        x = []
        for j in range(len(y[i])):
          idx = y_list.index(y[i][j])
          x.append(x_list[idx])
        x_group.append(x)
      return x_group

   def exp_function0(self,x,lamda,k,C):
       return lamda*(1+k*abs(x))*np.exp(-k*abs(x))+C

   def exp_function1(self,x,lamda,k):
       return lamda*(1+k*abs(x))*np.exp(-k*abs(x))

   def get_params(self,x,y):
       y_group = self.get_y(y)
       x_group = self.get_x(x,y,y_group)
       length = 0
       for i in range(len(x_group)):
         if len(x_group[i])> length :
            length = len(x_group[i])

       e = float('inf')
       for i in range(len(x_group)):
         if len(x_group[i]) == length: 
          if length == 2:
            param,_ = curve_fit(self.exp_function1, x_group[i], y_group[i], maxfev=100000000)
            y_fit = self.exp_function1(np.array(x_group[i]),param[0],param[1])
          elif length > 2:
            param,_ = curve_fit(self.exp_function0, x_group[i], y_group[i], maxfev=100000000)
            y_fit = self.exp_function0(np.array(x_group[i]),param[0],param[1],param[2])
          else:
            param_new = y_group[i]
            break
          if np.mean(abs(y_fit - np.array(y_group[i]))) < e:
            x_new = x_group[i]
            y_new = y_group[i]
            param_new = list(param)
            e = np.mean(abs(y_fit - np.array(y_group[i])))
       if len(param_new) == 1:
          param_new += [0.0,0.0]
       elif len(param_new) == 2:
          param_new += [0.0]
       
       return param_new

   def get_time(self):
       time_fit = []
       for i in range(len(self.E)):
          months = []
          for t in [16,46,75]:
            delt = self.holiday[i] - t
            months.append(delt)
            '''
            if abs(delt) < 4:
               months.append(0)
            elif delt < 0:
               months.append(delt+3)
            else:
               months.append(delt-3)
            '''
          time_fit.append(months)
       return time_fit                                    

   def fit_params(self,time):
       x0 = [];y0 = [];x1 = [];y1 = []
       for i in range(len(time)):
         for j in range(len(time[i])):
           if self.E[i][j] != float('inf'):
            if time[i][j] > 0:
              x0.append(abs(time[i][j]))
              y0.append(self.E[i][j])
            else:
              x1.append(abs(time[i][j]))
              y1.append(self.E[i][j])

       x0_new = [];y0_new = [];x1_new = [];y1_new = []
       for i in range(len(x0)):
          idx = x0.index(min(x0))
          x0_new.append(x0.pop(idx))
          y0_new.append(y0.pop(idx))
       for i in range(len(x1)):
          idx = x1.index(min(x1))
          x1_new.append(x1.pop(idx))
          y1_new.append(y1.pop(idx))
       param0 = self.get_params(x0_new,y0_new)
       param1 = self.get_params(x1_new,y1_new)
       return param0,param1
    

   def normalize_alphas(self):
       alphas_new = []
       time = self.get_time()
       param0,param1 = self.fit_params(time)
       for i in range(len(time)):
         alpha_new = []
         for j in range(len(time[i])):
           alpha_new.append(1+self.exp_function0(time[i][j],param0[0],param0[1],param0[2]))
         alphas_new.append(alpha_new)
       return alphas_new,param0,param1
    
   def update_dataset(self):

       data_new = []
       alphas_new,param0,param1 = self.normalize_alphas()
       for i in range(len(self.data)):
         if i%12 in [0,1,2]:
           data_new.append(self.data[i] /alphas_new[i//12][i%12])
         else:
           data_new.append(self.data[i])
       return data_new,param0,param1

   def update_y(self,y):
       alphas_new,param0,param1 = self.normalize_alphas()
       y1_new = [];y2_new = [];y3_new = []
       for i in range(1,len(y)):
         y1_new.append(y[i][0] * alphas_new[i][0])
         y2_new.append(y[i][1] * alphas_new[i][1])
         y3_new.append(y[i][2] * alphas_new[i][2])
       return np.mat(y1_new),np.mat(y2_new),np.mat(y3_new)

class predict:
   def __init__(self,data,data_res,param0,param1,W0,W1,W,holiday):
       self.data = data[len(data)-13:len(data)]
       self.data_res = data_res
       self.param0 = param0
       self.param1 = param1
       self.W0 = W0
       self.W1 = W1
       self.W = W
       self.t = holiday  #2019年春节

   def exp_function(self,x,lamda,k,C):
       return lamda*(1+k*abs(x))*math.exp(-k*abs(x))+C

   def get_time(self,t):
       time = []
       for i in [16,46,75]:
         delt = t - i
         time.append(delt)
         '''
         if abs(delt) < 4:
            time.append(0)
         elif delt < 0:
            time.append(delt+3)
         else:
            time.append(delt-3)
         '''
       return time    
   
   def get_W(self):
       l = len(self.data_res)
       if l < 3:l = 0
       for i in range(len(self.data_res),len(self.W)):
         self.W[i][0] = self.W[i-1][0] * self.W[i][0]
         self.W[i][1] = 1.0 - self.W[i][0]
       

   def preprocess(self):
       begin = (self.data).pop(0)
       for i in range(len(self.data)):
         if i in [0,1,2]:
           if i == 0:
             self.data[i] = begin * self.W0[i-1][0] + self.W0[i-1][1]
           else:
             self.data[i] = self.data[i-1] * self.W0[i-1][0] + self.W0[i-1][1]
       return self.data


   def predict_data(self):
       data = self.preprocess()
       self.get_W()
       time = self.get_time(self.t)
       y = [];alphas = [];result = []
       for i in range(12):
         if i == 0:
           y0 = self.W0[i-1][0] * data[i-1] + self.W0[i-1][1]
         else:
           y0 = self.W0[i-1][0] * y[-1] + self.W0[i-1][1]
         if i in [0,1,2]:
           if time[i] > 0:
             alpha = (1+self.exp_function(time[i],self.param0[0],self.param0[1],self.param0[2]))
           else:
             alpha = (1+self.exp_function(time[i],self.param1[0],self.param1[1],self.param1[2]))
           alphas.append(alpha)
         y1 = self.W1[i][0] * data[i] + self.W1[i][1]
         if len(self.data_res) > 3 and i < len(self.data_res):
            y.append(self.data_res[i])
         else:
            y.append(y0*self.W[i][0]+y1*self.W[i][1])
       
       for i in range(12):
         if i < len(self.data_res):
            result.append(int(self.data_res[i]))
         else:
           if i in [0,1,2]:
             result.append(int(y[i] * alphas[i]))
           else:
             result.append(int(y[i]))
       return result


READ = read_Excel(data_path = 'dataset.xlsx',sheet_index = 0,Sheet='Sheet1')
dataset = READ.read_data(row_index = list(range(2,27)),col_index = list(range(2,62)))
i = int(input('please input the index of the road to predict:'))
length = len(dataset[i])
data = dataset[i][0:length-length%12]
data_res = dataset[i][length-length%12:length]


data_old = data.copy()
#mode = input('train or predict?(t/p):')

j = 0
while(1):
 if j == 0:
   select = True
   MARKOV = markov(data_old,select)
   W0,W1,W,E = MARKOV.get_params()
   UPDATE = update(data_old,E)
   data_old,param0,param1 = UPDATE.update_dataset()
 else:
   select = False
   MARKOV = markov(data_old,select)
   W0,W1,W,E = MARKOV.get_params()
   break
 j += 1

PREDICT = predict(data,data_res,param0,param1,W0,W1,W,holiday = 40)
prediction = PREDICT.predict_data()
print('prediction',prediction)
print('finished')
