import numpy as np
import matplotlib.pyplot as plt

plt.axes()
line1=plt.Line2D((1, 3), (1, 1), lw = 2, marker = ".", color = "red")
line2=plt.Line2D((2, 3), (2, 2), lw = 2, marker = ".", color = "red")
line3=plt.Line2D((3, 3), (1, 2), lw = 2, marker = ".", color = "red")
line4=plt.Line2D((1, 2), (1, 2), lw = 2, marker = ".", color = "red")
plt.gca().add_line(line1)
plt.gca().add_line(line2)
plt.gca().add_line(line3)
plt.gca().add_line(line4)
plt.axis([0, 4, 0, 4])
plt.show()