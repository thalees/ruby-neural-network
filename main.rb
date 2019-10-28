require 'matrix'
require 'pry' # binding.pry
require_relative 'neural_network'

dataset = {
	inputs: [[1, 1], [1, 0], [0, 1],[0, 0]],
	outputs: [[0], [1], [1], [0]]
}

nn = NeuralNetwork.new(2, 3, 1)
nn.train([1, 0], [1])