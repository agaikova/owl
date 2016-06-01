"""
This is the file that organizes all the tests that I want to run. Here's a list:

For childhood = n (with n a number in the logorythmic array from 10 - 10 000), test the following childhood/glasses ratios: 1:10, 1:1, 10:1
...

Dump everything into a data folder. All data processing happens externally in using jupyter and the pandas and seaborn libraries in a jupyter notebook called data analysis

"""
import owl_parameter_search_intercepts

intercept_r = [0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0]

for seed in range(5):
    for intercept in intercept_r:
        owl_parameter_search_intercepts.Owl().run(right_intercept = intercept, seed = seed)
