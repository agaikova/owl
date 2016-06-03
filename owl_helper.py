"""
This is the file that organizes all the tests that I want to run. Here's a list:

For childhood = n (with n a number in the logorythmic array from 10 - 10 000), test the following childhood/glasses ratios: 1:10, 1:1, 10:1
...

Dump everything into a data folder. All data processing happens externally in using jupyter and the pandas and seaborn libraries in a jupyter notebook called data analysis

"""
import owl_parameter_search

glasses_values = [(1000/10), 1000, (1000*10)]

for seed in range(3):
    for glasses in glasses_values:
        owl_parameter_search.Owl().run(t_glasses_on = glasses, seed = seed)
