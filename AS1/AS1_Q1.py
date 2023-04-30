import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./AS1/Gamma4804.csv')
# a. What are the count, the mean, the standard deviation, the minimum, the 25th percentile, the median, the 75th percentile, and the maximum of the feature ? Please round your answers to the seventh decimal place.
rounded = round(dataset.describe(), 7)
print(rounded)


# b. Use the Shimazaki and Shinomoto (2007) method to recommend a bin width.  We will try  = 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, and 100.  What bin width would you recommend if we want the number of bins to be between 10 and 100 inclusively? You need to show your calculations to receive full credit.

def univariate(y):
   y_nvalid = 0
   y_min = None
   y_max = None
   y_mean = None
   for u in y:
      if (not np.isnan(u)):
         y_nvalid = y_nvalid + 1

         if (y_min is not None):
            if (u < y_min):
               y_min = u
         else:
            y_min = u

         if (y_max is not None):
            if (u > y_max):
               y_max = u
         else:
            y_max = u

         if (y_mean is not None):
            y_mean = y_mean + u
         else:
            y_mean = u

   if (y_nvalid > 0):
      y_mean = y_mean / y_nvalid

   return (y_nvalid, y_min, y_max, y_mean)
def shimazaki_criterion(y, d_list):

   number_bins = []
   matrix_boundary = []
   criterion_list = []
   leftb=[]
   rightb=[]
   y_nvalid, y_min, y_max, y_mean = univariate(y)
   print()

   if (y_nvalid <= 0):
      raise ValueError('There are no non-missing values in the data vector.')
   else:

      for delta in d_list:
         y_middle = delta * np.round(y_mean / delta)
         # print(f"Y_middle for {delta}",y_middle)
         # print(f"Y_max for {delta}",y_max)
         # print(f"Y_min for {delta}",y_min)
         n_bin_left = np.ceil((y_middle - y_min) / delta)
         n_bin_right = np.ceil((y_max - y_middle) / delta)
         y_low = y_middle - n_bin_left * delta

         list_boundary = []
         n_bin = n_bin_left + n_bin_right
         bin_index = 0
         bin_boundary = y_low
         for i in np.arange(n_bin):
            bin_boundary = bin_boundary + delta
            bin_index = np.where(y > bin_boundary, i+1, bin_index)
            list_boundary.append(bin_boundary)
         uvalue, ucount = np.unique(bin_index, return_counts = True)
         mean_ucount = np.mean(ucount)
         ssd_ucount = np.mean(np.power((ucount - mean_ucount), 2))
         criterion = (2.0 * mean_ucount - ssd_ucount) / delta / delta
         # print('Mean count', mean_ucount)
         # print('ssd_count', ssd_ucount)
         number_bins.append(n_bin)
         matrix_boundary.append(list_boundary)
         criterion_list.append(criterion)
         leftb.append(n_bin_left)
         rightb.append(n_bin_right)

        
   return(number_bins, matrix_boundary, criterion_list, leftb, rightb)

input_data = pd.read_csv('./AS1/Gamma4804.csv')
label = input_data['x']
deltas = [ 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100]
binsNo, matrix_boundary, cDelta, left, right = shimazaki_criterion(label, deltas)

result_df = pd.DataFrame(columns = ['Delta','Criterion Delta', 'Left Bin', 'Right Bin', 'No. of Bins'])
result_df['Delta'] = deltas
result_df['Criterion Delta'] = cDelta
result_df['No. of Bins'] = binsNo
result_df['Left Bin']=left
result_df['Right Bin']=right
print(result_df)


# c. Draw the density estimator using your recommended bin width answer in (b).  You need to label the graph elements properly to receive full credit.
def coordinates_his (inpVec, bWidth, bounBMin, BounBMax): 
    rowsnumber = inpVec.shape[0]
   #  print(rowsnumber)
    coordinates_histo = pd.DataFrame(columns=['MidPoint','Density'])
    midPoint = bounBMin + bWidth / 2
    iCoord = 0
    while (midPoint < BounBMax):
        m_u = (inpVec - midPoint) / bWidth
      #   print(m_u)
        hCoord = (m_u[(-0.5 < m_u) & (m_u <= 0.5)].count()) / (rowsnumber * bWidth)
        coordinates_histo.loc[iCoord] = [midPoint, hCoord]
        iCoord += 1
        midPoint += bWidth
      #   print(midPoint)
    return coordinates_histo

b = 5
histogramCoordinate = coordinates_his(label, b, 10, 200)
print(histogramCoordinate)

fig, ax = plt.subplots(1, 1, dpi = 100)
fig.set_size_inches(8, 6)
ax.set_title('Histogram Graph')
ax.step(histogramCoordinate['MidPoint'], histogramCoordinate['Density'], where = 'mid', label = 'b = ' + str(b), color = 'black')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('Estimated Density Value')
ax.set_xticks(np.arange(10,200,10.0))
ax.set_xlim(left = 10, right = 200)
ax.set_ylim(bottom = 0.0, top = 0.02)
ax.grid(axis = 'both', linestyle = ':')
ax.margins(y = 0.1)
plt.show()