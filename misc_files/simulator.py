"""
Particle simulator

Interaction:
	Button-2 + Drag : zoom
	Button-3 + Drag : camera movement
"""
from __future__ import division
from vpython import (color, display, curve, sphere, label)  # http://www.vpython.org/webdoc/visual/
from random import random
import time

# gnts ================================================================
WINDOW_TITLE = "Brownian Motion"
WINDOW_WIDTH = 640 + 4 + 4
WINDOW_HEIGHT = 480 + 24 + 4

CUBE_SIZE = 1.0  # container is a cube of CUBE_SIZE sides
CUBE_COLOR = color.white  # color of edges of container
CUBE_THICKNESS = 0.05  # radius of lines drawn on edges of cube

DIVISION_FACTOR = 0.5  # Factor two divide the cube into two parts
DIVISION_COLOR = color.green  # Division colour

PARTICLES_TOTAL = 10  # change this to have more or fewer particles
PARTICLES_SIZE = 0.2  # wildly exaggerated size of helium particle
PARTICLES_COLORS = [
    color.red,
    color.blue,
    color.yellow,
    color.cyan,
    color.magenta,
    color.green,
]

SLEEP_SECONDS = 0.25  # seconds to delay among each simulation dt
SAMPLING_RATE = 0.5  # dts to skip among each sampling

sigma = 10
# maximum displacement applied when computing the particles movement


# Window
window = display(
    title=WINDOW_TITLE,
    autocenter=1,
    autoscale=1,
    width=WINDOW_WIDTH,
    height=WINDOW_HEIGHT,
)

# Axis X
axisX = curve(
    pos=[(-sigma, 0, 0), (sigma, 0, 0)], color=color.white, radius=CUBE_THICKNESS
)


# Axis Y
axisY = curve(
    pos=[(0, -4 * sigma, 0), (0, 4 * sigma, 0)],
    color=color.white,
    radius=CUBE_THICKNESS,
)


# Particles
particles = []
streams = []

min = 1.1 * PARTICLES_SIZE
max = CUBE_SIZE - min

# Particles: creation
for i in range(PARTICLES_TOTAL):

    x = 0
    y = 0
    z = 0

    particle = sphere(
        pos=(x, y, z), radius=PARTICLES_SIZE, color=PARTICLES_COLORS[i % 6]
    )

    particles.append(particle)

    # Stream line
    streams.append(curve(pos=[(0, 0, 0)], color=color.green, radius=CUBE_THICKNESS))


# Run simulation
dt = 0

while True:

    # Update particles
    for i in range(PARTICLES_TOTAL):
        pos = particles[i].pos

        # Random walk
        dS = sigma * (random.random() - 0.5)

        pos[0] = dt * sigma
        pos[1] = pos[1] + dS
        pos[2] = 0

        particles[i].pos = pos
        streams[i].append(pos=pos)
        axisX.append(pos=(pos[0], 0, 0))

        newThickness = CUBE_THICKNESS * (dt * SAMPLING_RATE) + 1
        streams[i].radius = newThickness
        axisX.radius = newThickness * 0.5
        axisY.radius = newThickness * 0.5
        particles[i].radius = newThickness * 2.0

    time.sleep(SLEEP_SECONDS)

    dt = dt + 1
