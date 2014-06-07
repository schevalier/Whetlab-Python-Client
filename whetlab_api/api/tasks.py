# Returns the tasks set for a user
#
class Tasks():

	def __init__(self, client):
		self.client = client

	# Return the task set corresponding to user
	# '/alpha/tasks' GET
	#
	def get(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/alpha/tasks', body, options)

		return response

	# Creates a new task
	# '/alpha/tasks/' POST
	#
	# name - A short name for the task. Max 500 chars
	# description - A detailed description of the task
	# settings - The parameter specification of the parameters to tbe optimized
	def create(self, name, description, settings, options = {}):
		body = options['body'] if 'body' in options else {}
		body['name'] = name
		body['description'] = description
		body['settings'] = settings

		response = self.client.post('/alpha/tasks/', body, options)

		return response

