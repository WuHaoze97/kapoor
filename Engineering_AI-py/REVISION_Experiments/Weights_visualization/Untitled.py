# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: pytorch
# ---

# +
import numpy as np

# Set a seed for reproducibility (optional)
np.random.seed(40)

# Generate five random numbers between 0 and 1
random_numbers = np.random.rand(10)

# Sort the array in ascending order
sorted_numbers = np.sort(random_numbers)

# Round off the numbers to two decimal places
rounded_numbers = np.round(sorted_numbers, decimals=2)

print(rounded_numbers)

# -




