# Manipulate a result set indexed by its id
#
# id - Identifier of a result
class Result():

	def __init__(self, id, client):
		self.id = id
		self.client = client

	# Return a specific result indexed by id
	# '/alpha/results/:id/' GET
	#
	def get(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/alpha/results/' + self.id + '/', body, options)

		return response

	# Delete the result instance indexed by id
	# '/alpha/results/:id/' DELETE
	#
	def delete(self, options = {}):
		body = options['body'] if 'body' in options else {}

		response = self.client.delete('/alpha/results/' + self.id + '/', body, options)

		return response

	# Update a specific result indexed by id
	# '/alpha/results/:id/' PATCH
	#
	# variables - The result list of dictionary objects with updated fields.
	# experiment - Experiment id
	# userProposed - userProposed
	# description - description
	# id - <no value>
	def update(self, variables, experiment, userProposed, description, id, options = {}, **kwargs):
		body = options['body'] if 'body' in options else {}

		body['variables']    = variables
		body['experiment']   = experiment
		body['userProposed'] = userProposed
		body['description']  = description
		body['id']           = id

		response = self.client.patch('/alpha/results/' + self.id + '/', body, options)

		return response

	# Replace a specific result indexed by id. To be used instead of update if HTTP patch is unavailable
	# '/alpha/results/:id/' PUT
	#
	# variables - The result list of dictionary objects with updated fields.
	# experiment - Experiment id
	# userProposed - userProposed
	# description - description
	# id - <no value>
	def replace(self, variables, experiment, userProposed, description, id, options = {}):
		body = options['body'] if 'body' in options else {}

		body['variables']    = variables
		body['experiment']   = experiment
		body['userProposed'] = userProposed
		body['description']  = description
		body['id']           = id

		response = self.client.put('/alpha/results/' + self.id + '/', body, options)

		return response

