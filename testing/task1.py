import math
import random
from scipy.stats import chi2

N = 100
interval = (0, 100)
K = 10
ALPHA = 0.01


def mean(intervals, frequencies):
    total = 0
    for i in range(len(intervals)):
        left, right = intervals[i][0], intervals[i][1]
        x_i = (left + right) / 2
        total += (x_i * frequencies[i])
    mean_v = total / N
    return mean_v


def predicted(intervals, frequencies):
    predicted_nums = [0] * len(intervals)
    mean_v = mean(intervals, frequencies)
    L = 1 / mean_v

    for i in range(len(predicted_nums)):
        (left, right) = intervals[i]
        prob = math.exp(-L * left) - math.exp(-L * right)
        expected_freq = prob * N
        predicted_nums[i] = expected_freq
    return predicted_nums


def solve():
    intervals = [0] * K
    frequencies = [0] * K
    random_nums = [0] * N

    interval_width = (interval[0] + interval[1]) / K
    for i in range(K):
        left = i * interval_width
        right = left + interval_width
        intervals[i] = (left, right)

    # print(intervals)
    for i in range(len(random_nums)):
        random_nums[i] = random.randint(interval[0], interval[1])

    # print(random_nums)
    for i in range(len(random_nums)):
        num = random_nums[i]
        len_intervals = len(intervals)
        for j in range(len_intervals):
            (left, right) = intervals[j]
            if j < len_intervals - 1 and left <= num < right:
                frequencies[j] = frequencies[j] + 1
                break
            elif j == len_intervals - 1 and left <= num <= right:
                frequencies[j] = frequencies[j] + 1
                break
    # print(frequencies)

    predicted_values = predicted(intervals, frequencies)
    # print(predicted_values)

    X2 = 0
    for i in range(len(frequencies)):
        if predicted_values[i] > 0:
            x = (frequencies[i] - predicted_values[i]) ** 2 / predicted_values[i]
            X2 += x
    # print(X2)

    degrees_freedom = lambda k, s: k - (s + 1)
    # print(degrees_freedom(len(intervals), 1))

    p_value = chi2.cdf(X2, degrees_freedom(len(intervals), 1))
    print("p_value =", p_value)

    if p_value >= ALPHA:
        print("Гипотезу отклонить нельзя")
    else:
        print("Гипотеза отклоняется.")


solve()