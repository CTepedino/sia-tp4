import math

def min_max(data, a=0, b=1):
    data_max = max(data)
    data_min = min(data)
    return [((x - data_min)/(data_max-data_min))*(b-a)+a for x in data]

def standardize(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)

def unit_length_scaling(data):
    norm_2 = math.sqrt(sum(x**2 for x in data))
    return [x/norm_2 for x in data]

