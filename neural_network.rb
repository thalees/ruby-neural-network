require 'matrix'
require 'pry' # binding.pry
require_relative 'calculator'

class NeuralNetwork

  def initialize(input, hidden, output)
    @input = input
    @hidden = hidden
    @output = output
    
    @bias_input_hidden = Matrix.build(hidden, 1) { rand }
    @bias_hidden_output = Matrix.build(output, 1) { rand }

    @weigth_input_hidden = Matrix.build(hidden, input) { rand }
    @weigth_hidden_output = Matrix.build(output, hidden) { rand }
    
    @calculator = Calculator.new
  end
  
  def train(input_array)
    # Feedforward
    hidden = input_to_hidden(input_array)
    output = hidden_to_output(hidden)
  end

  def input_to_hidden(input_array)
    input_matrix = Matrix.column_vector(input_array)
    hidden = @weigth_input_hidden * input_matrix
    @calculator.sigmoid(hidden + @bias_input_hidden)
  end

  def hidden_to_output(hidden_matrix)
    output = @weigth_hidden_output * hidden_matrix
    @calculator.sigmoid(output + @bias_hidden_output)
  end
end