# -*- coding: utf-8 -*-
"""Project 1

Least squares
"""
from helpers import*
import numpy as np
from least_squares import*


[y, x] = load_and_build_data()
print(type(x), type(y))
[w, mse] = least_squares(y, x)

print(mse)