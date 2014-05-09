# Ask the server to propose a new set of parameters to run the next experiment
#
# taskid - Identifier of corresponding task
class Suggest():

	def __init__(self, taskid, client):
		self.taskid = taskid
		self.client = client

	# Ask the server to propose a new set of parameters to run the next experiment
	# '/alpha/tasks/:taskid/suggest/' POST
	#
	def go(self, options = {}):
		body = options['body'] if 'body' in options else {}

		response = self.client.post('/alpha/tasks/' + self.taskid + '/suggest/', body, options)

		return response

