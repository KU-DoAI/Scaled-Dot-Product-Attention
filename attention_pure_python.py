# Scaled Dot-Product Attention using pure Python lists
# --------------------------------------------------
# Implements scaled dot-product attention from scratch using only Python lists.
# Demonstrates unmasked, causal masked, and padding masked attention, and visualizes
# attention weights as heatmaps using matplotlib.

import math
import matplotlib.pyplot as plt

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

M = [[1, 2], [3, 4], [5, 6]]
print(transpose(M))
