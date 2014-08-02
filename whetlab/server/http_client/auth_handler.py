# AuthHandler takes care of devising the auth type and using it
class AuthHandler():

	URL_SECRET = 2
	URL_TOKEN = 3

	def __init__(self, auth):
		self.auth = auth

	# Calculating the Authentication Type
	def get_auth_type(self):

		if 'client_id' in self.auth and 'client_secret' in self.auth:
			return self.URL_SECRET

		if 'access_token' in self.auth:
			return self.URL_TOKEN

		return -1

	def set(self, request):
		if len(self.auth.keys()) == 0:
			return request

		auth = self.get_auth_type()
		flag = False

		if auth == self.URL_SECRET:
			request = self.url_secret(request)
			flag = True

		if auth == self.URL_TOKEN:
			request = self.url_token(request)
			flag = True

		if not flag:
			raise StandardError("Unable to calculate authorization method. Please check")

		return request

	# OAUTH2 Authorization with client secret
	def url_secret(self, request):
		request['params']['client_id'] = self.auth['client_id']
		request['params']['client_secret'] = self.auth['client_secret']
		return request

	# OAUTH2 Authorization with access token
	def url_token(self, request):
		request['params']['access_token'] = self.auth['access_token']
		return request

