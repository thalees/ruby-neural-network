require 'matrix'
require 'pry' # binding.pry
require_relative 'neural_network'

train = true

dataset = {
	inputs: [[1, 1], [1, 0], [0, 1],[0, 0]],
	outputs: [[0], [1], [1], [0]]
}

nn = NeuralNetwork.new(2, 3, 1)

while(train)
	for i in 0..10000
		index = rand(0...4)
		nn.train(dataset[:inputs][index], dataset[:outputs][index])
	end
	
	if (nn.predict([0, 0])[0].first < 0.04 && nn.predict([1, 0])[0].first > 0.8)
		train = false;
		puts "terminou"
	end
end

puts nn.predict([1, 0])[0].first # true