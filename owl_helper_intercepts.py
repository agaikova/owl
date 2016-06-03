"""
This is the file that organizes all the tests that I want to run. 

"""
import owl_parameter_search_intercepts

intercept_r = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]

for seed in range(2):
    for intercept in intercept_r:
        owl_parameter_search_intercepts.Owl().run(right_intercept = intercept, seed = seed)
