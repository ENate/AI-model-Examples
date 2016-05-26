'''
Created on May 3, 2016

Modified by: Nate
'''
"""
Sparse Neural Network Diagram
----------------------
"""
#This code is modified for personal use. Kindly contact 
# message4nath@gmail.com, for comments about this updated version.
#For information on the original implementation, please refer to:
# Author: Jake VanderPlas <vanderplas@astro.washington.edu>
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
import numpy as np
from matplotlib import pyplot as plt

fig = plt.figure(facecolor='w')
ax = fig.add_axes([0, 0, 1, 1],
                  xticks=[], yticks=[])
plt.box(False)
circ = plt.Circle((1, 1), 2)
radius = 0.1

arrow_kwargs = dict(head_width=0.05, fc='black')

# function to draw arrows
def draw_connecting_arrow(ax, circ1, rad1, circ2, rad2):
    theta = np.arctan2(circ2[1] - circ1[1],
                       circ2[0] - circ1[0])

    starting_point = (circ1[0] + rad1 * np.cos(theta),
                      circ1[1] + rad1 * np.sin(theta))

    length = (circ2[0] - circ1[0] - (rad1 + 1.4 * rad2) * np.cos(theta),
              circ2[1] - circ1[1] - (rad1 + 1.4 * rad2) * np.sin(theta))

    ax.arrow(starting_point[0], starting_point[1],
             length[0], length[1], **arrow_kwargs)


# function to draw circles
def draw_circle(ax, center, radius):
    circ = plt.Circle(center, radius)#, fc='none', lw=2)
    ax.add_patch(circ)

x1 = -2
x2 = 0
x3 = 2
y3 = 0
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Import the required data from text file
matrix0 = np.loadtxt('ath/to/file')
hiddenOut = np.loadtxt('path/to/text/file')
#convert to binary matrix, with 1 for nonzero values, 0 otherwise!
hidden2Out1 = np.where(hiddenOut!=0,1,0)
new_matrix = np.where(matrix0 != 0,1,0)
hidden2Out0= hidden2Out1[::-1]
#hidden2Out00=np.flipud(hidden2Out0)
hidden2Out = hidden2Out0.T
#Arrange resulting binary matrix to suit the drawing as specified in the program.
weights11 = np.fliplr(new_matrix)
weights10 = np.flipud(weights11)
Inputs2hidden = weights10.T
size_xy = weights10.shape
#Arrange the neurons connecting hidden to outputs

# draw circles
for i, y1 in enumerate(np.linspace(1.5, -1.5, size_xy[0])): #hidden nodes neurons
    draw_circle(ax, (x1, y1), radius)
    ax.text(x1 - 0.9, y1, 'Input #%i' % (i + 1),ha='right', va='center', fontsize=16)
    draw_connecting_arrow(ax, (x1 - 0.9, y1), 0.1, (x1, y1), radius)

for y2 in np.linspace(-2, 2, size_xy[1]): #number of inputs
    draw_circle(ax, (x2, y2), radius)

draw_circle(ax, (x3, y3), radius)
ax.text(x3 + 0.8, y3, 'Output', ha='left', va='center', fontsize=16)
draw_connecting_arrow(ax, (x3, y3), radius, (x3 + 0.8, y3), 0.1)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# draw connecting arrows
#for y1 in np.linspace(-1.5, 1.5, 7):
#for y2 in np.linspace(-2, 2, 4):
xyValues = np.linspace(-1.5, 1.5, size_xy[0])
xyValues1 = np.linspace(-2, 2, size_xy[1])
print '==================================================='
print 'Optimal Sparse Neural Network weight matrices.'
print '==================================================='
print 'Weights connecting inputs to hidden neurons:'
print '==================================================='
print new_matrix
print '===================================================='

print 'Weights connecting hidden to output neurons:'
print '===================================================='
print hidden2Out1
print '===================================================='
# The plot is performed in the reverse direction......
for y11 in range(0,len(xyValues)):
    for y22 in range(0,len(xyValues1)):
        if Inputs2hidden[y22][y11] != 0:
            draw_connecting_arrow(ax, (x1, xyValues[y11]), radius, (x2, xyValues1[y22]), radius)

#for y2 in np.linspace(-2, 2, 4):
for y22 in range(0,len(xyValues1)):
    if hidden2Out[y22] != 0:
        draw_connecting_arrow(ax, (x2, xyValues1[y22]), radius, (x3, y3), radius)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Add text labels
plt.text(x1, 2.7, "Input\nLayer", ha='center', va='top', fontsize=16)
plt.text(x2, 2.7, "Hidden Layer", ha='center', va='top', fontsize=16)
plt.text(x3, 2.7, "Output\nLayer", ha='center', va='top', fontsize=16)

ax.set_aspect('equal')
plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.show()
