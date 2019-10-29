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
    
    @learning_rate = 0.1
    
    @calculator = Calculator.new
  end
  
  def train(input_array, target)
    # Feedforward
    hidden = input_to_hidden(input_array)
    output = hidden_to_output(hidden)

    # Backpropagation
    # Output -> Hidden
    expected = Matrix.column_vector(target)
    output_error = expected - output

    d_output = @calculator.dsigmoid(output)

    hidden_T = hidden.transpose

    gradient = @calculator.hadamard(d_output, output_error) * @learning_rate
    
    # Adjust Bias Output -> Hidden
    @bias_hidden_output = @bias_hidden_output + gradient
    
    # Adjust Weigths Output -> Hidden
    weigths_ho_delta = gradient * hidden_T
    @weigth_hidden_output = @weigth_hidden_output + weigths_ho_delta
    
    # Hidden -> Input
    weigths_ho_T = @weigth_hidden_output.transpose

    hidden_error = weigths_ho_T * output_error
    d_hidden = @calculator.dsigmoid(hidden)

    input_T = Matrix.column_vector(input_array).transpose
  
    gradient_H = @calculator.hadamard(d_hidden, hidden_error) * @learning_rate
    
    # Adjust Bias Output -> Hidden
    @bias_input_hidden = @bias_input_hidden + gradient_H
    
    # Adjust Weigths Hidden -> Input
    weigths_ih_deltas = gradient_H * input_T
    @weigth_input_hidden = @weigth_input_hidden + weigths_ih_deltas
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
  
  def predict(array)
    # Input -> Hidden
    input = Matrix.column_vector(array)

    hidden = @weigth_input_hidden * input
    hidden = @calculator.sigmoid(hidden + @bias_input_hidden)

    # Hidden -> Output
    output = @weigth_hidden_output * hidden
    @calculator.sigmoid(output + @bias_hidden_output).to_a
  end
end