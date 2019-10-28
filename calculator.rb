require 'matrix'

class Calculator
	EULER = 2.7182818284

	def sigmoid(matrix)
		matrix.map! do |value|
			1/(1+EULER**(-value))
		end
	end
end