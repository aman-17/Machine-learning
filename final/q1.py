import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt

df = pd.read_csv('./final/FinalQ1.csv',
                             delimiter=',', usecols = ['x'])


def get_coordinates(x, h, min_value, max_value):
    midpoints_list = []
    density_list = []
    count = 0
    mid_point = min_value + h / 2
    while(mid_point < max_value):
        v = (x - mid_point) / h
        for i in range(len(v)):
            if v[i] > -0.5 and v[i] <= 0.5:
                count += 1
        density = count / (len(x) * h)
        count = 0
        midpoints_list.append(round(mid_point, 2))
        density_list.append(round(density, 4))
        mid_point += h
    return midpoints_list, density_list


IQR = (np.percentile(df['x'], 75)) - (np.percentile(df['x'], 25))
print("Interquartile Range is: ", IQR)
N = len(df['x'])
bin_width = 4 * IQR * (N ** (-1/3))
print("Bin-width for hisogram of x is: ", bin_width)

print()

min_value = min(df['x'])
max_value = max(df['x'])
print("Minimum value of x is: ", min_value)
print("Maximum value of x is: ", max_value)

print()

print("Value of a is: ", mt.floor(min_value))
print("Value of b is: ", mt.ceil(max_value))

print()

print("for h = 2:")
mid_point, density = get_coordinates(df['x'], 3, min_value-1, max_value)
print('Mid-points', end='  ')
print('Density')
for i in range(len(mid_point)):
    print(mid_point[i], end='       ')
    print(density[i])
    



