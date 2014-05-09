# Returns the variables set for a user
#
class Variables():

	def __init__(self, client):
		self.client = client

	# Return the variables set corresponding to user
	# '/alpha/variables' GET
	#
	def get(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/alpha/variables', body, options)

		return response

