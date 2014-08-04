# Ask the server to propose a new set of parameters to run the next experiment
#
# exptid - Identifier of corresponding experiment
class Suggest():

	def __init__(self, exptid, client):
		self.exptid = exptid
		self.client = client

	# Ask the server to propose a new set of parameters to run the next experiment
	# '/alpha/experiments/:exptid/suggest/' POST
	#
	def go(self, options = {}):
		body = options['body'] if 'body' in options else {}

		response = self.client.post('/alpha/experiments/' + self.exptid + '/suggest/', body, options)

		return response

