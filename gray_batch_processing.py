import numpy as np
import math
import pylab as pl
import scipy.signal as signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import xlrd
import xlwt

#读取数据
def read_Excel(data_path,Sheet,sheet_index, row_index, col_index):
     dataset = []
     workbook = xlrd.open_workbook(data_path)  
     sheet_name = workbook.sheet_names()[sheet_index]
     sheet = workbook.sheet_by_name(Sheet)   
     for j in row_index: 
         data = []
         for i in col_index:
            data.append(sheet.cell(j,i).value)
         dataset.append(data)
     return dataset


def write_data(data):
    f = xlwt.Workbook() 
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) 
  
    for i in range(len(data)):
      for j in range(len(data[i])):
        sheet1.write(i,j,float(data[i][j]))

    f.save('text2.xlsx')

class grey_prediction:
  def __init__(self,dataset,n):
      self.dataset = dataset
      self.n = n

  def prediction(self,data,m):
      x0 = []
      x1 = []
      sum = 0.0
      for i in range(m):
         x0.append(data[i])
         sum += data[i]
         x1.append(sum)
      if len(data) != 1:
         B = []
         for i in range(m-1):
           B.append([-1/2*(x1[i]+x1[i+1]),1])
         y = x0.copy() 
         y.pop(0)
         B = np.mat(B)
         y = np.mat(y).T
         ap=(((B.T)*B).I)*(B.T)*y
         a=ap[0,0]
         u=ap[1,0]
         result = []
         for i in range(m + 1):
            result.append(int((x0[0] - u/a)* np.exp(-i*a)+u/a))
      else : raise ValueError("the number of prepared data must be larger than 1.")
      return result

  def prepare_data(self,data):
       months = []
       k = 0
       while k <12 :
         month = []
         for i in range(len(data)):
            if i % 12 == k:
              month.append(data[i])
         months.append(month)
         k += 1
       return months

  def get_result(self,data):
       grey_result = []
       total_result = []
       months = self.prepare_data(data)
       for i in range(self.n):
          results = [0]
          result = self.prediction(months[i],len(months[i]))
          for j in range(len(result)-1):
             delt = result[j+1] - result[j]
             results.append(delt)
          grey_result.append(results[-1])
          total_result.append(results)
       return grey_result,total_result

  def pre_process(self):
      results = []
      for i in range(len(self.dataset)):
          grey_result,_ = self.get_result(self.dataset[i])
          results.append(grey_result)
      return results




dataset = read_Excel(data_path = 'dataset.xlsx',Sheet ='Sheet1' ,sheet_index = 0, row_index = list(range(2,28)), col_index = list(range(2,62)))
GREY_PREDICTION = grey_prediction(dataset ,12)
data = GREY_PREDICTION.pre_process()
write_data(data)
print('prediction:',data)
print('finished')
