require 'matrix'

class Calculator
	EULER = 2.7182818284

	def sigmoid(matrix)
		matrix.map! do |value|
			1/(1+EULER**(-value))
		end
	end

	def dsigmoid(matrix)
		matrix.map! do |value|
			value * (1 - value)
		end
	end

	def hadamard(a, b)
		matrix = Matrix.build(a.row_size, a.column_size) { nil }
	
		matrix.each_with_index do |elm, row, col|
		  matrix[row, col] = a[row, col] * b[row, col]
		end
		
		matrix
	end
end