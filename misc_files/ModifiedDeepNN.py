"""
Created on Apr 20, 2017

@author: NATH
#Modified version of a deep neural network diagram code
#Partly obtained from a discussion on www.stackoverflow.com- and
#modified for personal use for my thesis.
"""
from math import cos, sin, atan

from matplotlib import pyplot
import numpy as np

# Define the number of neurons in each layer
class Neuron:
    """Define neuron position."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self):
        """Draw to be called."""
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True)
        pyplot.gca().add_patch(circle)
    
    def layer_print(self):
        """Prints the individual layers."""


# Define class for each layer with number of connecting weights, number of neurons,
# And initialize. A provision is made to cater for the horizontal
# as well as vertical distances between layers. It also defines
# a function to draw the connecting weights, the individual neurons, too.
class Layer:
    """Defines the Layer object."""
    def __init__(self, network, number_of_neurons, weights, distance):
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.weights = weights
        self.horizontal_distance_between_neurons = distance

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return (
            self.horizontal_distance_between_neurons
            * (number_of_neurons_in_widest_layer - number_of_neurons)
            / 2
        )

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, previous_network):
        if len(previous_network.layers) > 0:
            return previous_network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, linewidth):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = neuron_radius * sin(angle)
        y_adjustment = neuron_radius * cos(angle)
        line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
        line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)
        line = pyplot.Line2D(line_x_data, line_y_data, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self):
        for this_layer_neuron_index in range(len(self.neurons)):
            neuron = self.neurons[this_layer_neuron_index]
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron_index in range(
                    len(self.previous_layer.neurons)
                ):
                    previous_layer_neuron = self.previous_layer.neurons[
                        previous_layer_neuron_index
                    ]
                    weight = self.previous_layer.weights[
                        this_layer_neuron_index, previous_layer_neuron_index
                    ]
                    self.__line_between_two_neurons(
                        neuron, previous_layer_neuron, weight
                    )


# Appends the layers to each other, add neurons and defines the
# drawing area. It also scales the axes and provides labels.
class NeuralNetwork:
    """Initialize."""
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons, weights=None):
        """Define num of neurons and add layer."""
        layer = Layer(self, number_of_neurons, weights)
        self.layers.append(layer)

    def draw(self):
        """Draw network is called."""
        for layer in self.layers:
            layer.draw()
        pyplot.axis("scaled")
        pyplot.axis("off")
        pyplot.title("Neural Network architecture", fontsize=12)
        pyplot.show()


# Execution starts here. Provide the weights as a matrix, or define weight entries from
# a file.
if __name__ == "__main__":
    # vertical_distance_between_layers = 6
    VERTICAL_DISTANCE_BETWEEN_LAYERS = 6
    # horizontal_distance_between_neurons = 2
    HORIZONTAL_DISTANCE_BETWEEN_NEURONS = 2
    NEURON_RADIUS = 0.5
    # neuron_radius = 0.5
    # number_of_neurons_in_widest_layer = 7
    NUMBER_OF_NEURONS_IN_WIDEST_LAYER = 7
    network = NeuralNetwork()
    # weights to convert from 10 outputs to 4 (decimal digits to their binary representation)
    # Matrices with rows as no. of weights, and columns as no. of neurons
    input_weights = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ]
    )
    colWeights1 = input_weights.shape
    h1h2_weights = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    colWeights2 = h1h2_weights.shape
    h2Outweights = np.array([[1, 0, 1]])
    colWeights3 = h2Outweights.shape
    network.add_layer(colWeights1[1], input_weights)
    network.add_layer(colWeights2[1], h1h2_weights)
    network.add_layer(colWeights3[1], h2Outweights)
    network.add_layer(1)
    network.draw()
