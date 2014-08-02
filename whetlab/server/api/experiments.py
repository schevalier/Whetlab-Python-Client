# Returns the experiments set for a user
#
class Experiments():

	def __init__(self, client):
		self.client = client

	# Return the experiments set corresponding to user
	# '/alpha/experiments/' GET
	#
	def get(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/alpha/experiments/', body, options)

		return response

	# Create a new experiment and get the corresponding id
	# '/alpha/experiments/' POST
	#
	# name - The name of the experiment to be created.
	# description - A detailed description of the experiment
	# user - The user id of this user
	def create(self, name, description, user, options = {}):
		body = options['body'] if 'body' in options else {}
		body['name'] = name
		body['description'] = description
		body['user'] = user

		response = self.client.post('/alpha/experiments/', body, options)

		return response

