# Return user list
#
class Users():

	def __init__(self, client):
		self.client = client

	# <no value>
	# '/users' GET
	#
	def getusers(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/users', body, options)

		return response

