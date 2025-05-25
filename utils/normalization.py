import math


def min_max(data, a=0, b=1):
    data_max = max(data)
    data_min = min(data)
    return [((x - data_min)/(data_max-data_min))*(b-a)+a for x in data]

def mean(data):
    return sum(data)/len(data)

def standard_deviation(data):
    data_mean = mean(data)
    return math.sqrt((sum(x - data_mean for x in data))/len(data))

def standardize(data):
    return (data - mean(data))/standard_deviation(data)

def unit_length_scaling(data):
    norm_2 = math.sqrt(sum(x**2 for x in data))
    return [x/norm_2 for x in data]

