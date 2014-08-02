# Manipulate the results set for an experiment given filters
#
class Results():

	def __init__(self, client):
		self.client = client

	# Return a result set corresponding to an experiment
	# '/alpha/results' GET
	#
	def get(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/alpha/results', body, options)

		return response

	# Add a user created result
	# '/alpha/results/' POST
	#
	# variables - The result list of dictionary objects with updated fields.
	# experiment - Experiment id
	# userProposed - userProposed
	# description - description
	def add(self, variables, experiment, userProposed, description, options = {}):
		body = options['body'] if 'body' in options else {}

		body['variables']    = variables
		body['experiment']   = experiment
		body['userProposed'] = userProposed
		body['description']  = description

		response = self.client.post('/alpha/results/', body, options)

		return response

