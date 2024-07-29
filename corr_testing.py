'''
CORR TESTING
by Josha Paonaskar

Correlation testing to validate and analylize batch methods

Resources:

'''

import benchmark

benchmark.batch_size_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40], win_size=12, search_size=24, test_window=2.0)