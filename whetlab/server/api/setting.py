# Manipulate an experimental settings object
#
class Setting():

	def __init__(self, client):
		self.client = client

	# Set a setting corresponding to an experiment
	# '/alpha/settings/' POST
	#
	# name - The name of the variable.
	# type - The type of variable. One of int,float,etc.
	# min - Minimum value for the variable
	# max - Maximum value for the variable
	# size - Vector size for the variable
	# units - What units is the variable in?
	# experiment - The experiment associated with this variable
	# scale - The scale of the units associated with this variable
	# isOutput - Is this variable an output of the experiment
	def set(self, name, type, min, max, size, units, experiment, scale, isOutput, options = {}):
		body = options['body'] if 'body' in options else {}
		body['name'] = name
		body['type'] = type
		body['min'] = min
		body['max'] = max
		body['size'] = size
		body['units'] = units
		body['experiment'] = experiment
		body['scale'] = scale
		body['isOutput'] = isOutput

		response = self.client.post('/alpha/settings/', body, options)

		return response

