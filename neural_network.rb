require 'matrix'
require 'pry' # binding.pry

class NeuralNetwork

  def initialize(input, hidden, output)
    @input = input
    @hidden = hidden
    @output = output
    
    @bias_input_hidden = Matrix.build(hidden, 1) { rand }
    @bias_hidden_output = Matrix.build(output, 1) { rand }

    @weigth_input_hidden = Matrix.build(hidden, input) { rand }
    @weigth_hidden_output = Matrix.build(output, hidden) { rand }
  end
end