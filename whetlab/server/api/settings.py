# Returns the settings config for an experiment
#
class Settings():

	def __init__(self, client):
		self.client = client

	# Return the settings corresponding to the experiment.
	# '/alpha/settings/' GET
	#
	# experiment - Experiment id to filter by.
	def get(self, experiment, options = {}):
		body = options['query'] if 'query' in options else {}
		body['experiment'] = experiment

		response = self.client.get('/alpha/settings/', body, options)

		return response

