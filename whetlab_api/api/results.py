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
	# task - Task id
	# userProposed - userProposed
	# description - description
	# runDate - <no value>
	def add(self, task, userProposed, description, runDate, options = {}):
		body = options['body'] if 'body' in options else {}
		body['task'] = task
		body['userProposed'] = userProposed
		body['description'] = description
		body['runDate'] = runDate

		response = self.client.post('/alpha/results/', body, options)

		return response

