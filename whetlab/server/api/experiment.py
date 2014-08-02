# Manipulate the experiment indexed by id.
#
# id - Identifier of corresponding experiment
class Experiment():

	def __init__(self, id, client):
		self.id = id
		self.client = client

	# Return the experiment corresponding to id.
	# '/alpha/experiments/:id/' GET
	#
	def get(self, options = {}):
		body = options['query'] if 'query' in options else {}

		response = self.client.get('/alpha/experiments/' + self.id + '/', body, options)

		return response

	# Delete the experiment corresponding to id.
	# '/alpha/experiments/:id/' DELETE
	#
	def delete(self, options = {}):
		body = options['body'] if 'body' in options else {}

		response = self.client.delete('/alpha/experiments/' + self.id + '/', body, options)

		return response

