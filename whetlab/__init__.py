import os
import ast
import tempfile
import ConfigParser
import collections
import numpy as np
import server
import time
import re
import functools
import requests
from whetlab.server.error.client_error import *

def catch_exception(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            raise RuntimeError('Unable to reach the server. Either the server is experiencing difficulties or your internet connection is down.')
        except server.error.ClientError as e:
            if e.message == "Unable to understand the content type of response returned by request responsible for error":
                raise RuntimeError('The server is currently busy, please try again shortly.')
            else:
                raise e
    return func


INF_PAGE_SIZE = 1000000

DEFAULT_API_URL = 'https://www.whetlab.com'

supported_properties = set(['min','max','size','type','options'])
required_properties = {
    'float':set(['min','max']),
    'integer':set(['min','max']),
    'enum':set(['options'])
}
default_values = {
    'float':{
        'size':1,
        'scale':'linear',
        'units':'Reals',
    },
    'integer':{
        'size':1,
        'scale':'linear',
        'units':'Integers',
    },
    'enum':{
        'size':1
    },
    'type':'float'
}
legal_values = {
    'size':set(range(25)),
    'scale':set(['linear','log']),
    'type':set(['float', 'integer', 'enum'])
}

python_types = {'float':float,'integer':int,'enum':str}

outcome_supported_properties = set(['name'])
outcome_required_properties = set(['name'])
outcome_default_values = {'min':-100.,
              'max':100.,
              'size':1,
              'scale':'linear',
              'units':'Reals',
              'type':'float'}
outcome_legal_values = {'size':set([1]),
            'scale':set(['linear']),
            'type':set(['float'])}

@catch_exception
def _reformat_float(rest_param):
    """
    Convert float parameter description in REST-server format to internal client format.

    :param rest_param: Parameter description obtained from REST server.
    :type rest_param: dict
    :return: Parameter description in interal client format.
    :rtype: dict
    """

    type  = rest_param['type']
    min   = rest_param['min']
    max   = rest_param['max']
    size  = rest_param['size']
    units = rest_param['units']
    scale = rest_param['scale']

    return {'type':type,'min':min,'max':max,
            'size':size,'units':units,'scale':scale}

@catch_exception
def _reformat_integer(rest_param):
    """
    Convert integer parameter description in REST-server format to internal client format.

    :param rest_param: Parameter description obtained from REST server.
    :type rest_param: dict
    :return: Parameter description in interal client format.
    :rtype: dict
    """

    type  = rest_param['type']
    min   = rest_param['min']
    max   = rest_param['max']
    size  = rest_param['size']
    units = rest_param['units']
    scale = rest_param['scale']

    return {'type':type,'min':min,'max':max,
            'size':size,'units':units,'scale':scale}

@catch_exception
def _reformat_enum(rest_param):
    """
    Convert enum parameter description in REST-server format to internal client format.

    :param rest_param: Parameter description obtained from REST server.
    :type rest_param: dict
    :return: Parameter description in interal client format.
    :rtype: dict
    """

    type       = rest_param['type']
    options    = rest_param['options']
    size       = rest_param['size']

    return {'type':type,'options':options,'size':size}

@catch_exception
def _validate_integer(name, properties):
    """
    Validate that integer parameter description is valid.

    :param name: Name of parameter
    :type name: str
    :param name: Properties of the parameter
    :type name: dict
    """

    # Check if required properties are present
    for property in required_properties['integer']:
        if property not in properties:
            raise ValueError("Parameter '" +name+ "': property '" + property + "' must be defined")

    # Add default parameters if not present
    for property, default in default_values['integer'].iteritems():
        if property not in properties:
            properties[property] = default

    # Finite bounds
    if not np.isfinite(properties['min']) or not np.isfinite(properties['max']):
        raise ValueError("Parameter '" + name + "': 'min' and 'max' must be finite.")    
    if (properties['min'] < -1e32 or properties['max'] < -1e32 or properties['min'] < -1e32 or 
        properties['min'] > 1e32  or properties['max'] > 1e32):
        raise ValueError("Parameter '" + name + "': 'min' and 'max' must be finite and between -1e32 and 1e32.")

    # Check compatibility of properties
    if not (properties['min'] < properties['max']):
        raise ValueError("Parameter '" + name + "': 'min' should be smaller than 'max'")

    if np.mod(properties['min'],1) != 0 : raise ValueError("Parameter '" + name + "': 'min' should be an integer")
    if np.mod(properties['max'],1) != 0 : raise ValueError("Parameter '" + name + "': 'max' should be an integer")

    for property, legals in legal_values.iteritems():
        if properties[property] not in legals:
            raise ValueError("Parameter '" +name+ "': invalid value for property '" + property+"'")

@catch_exception
def _validate_float(name, properties):
    """
    Validate that float parameter description is valid.

    :param name: Name of parameter
    :type name: str
    :param name: Properties of the parameter
    :type name: dict
    """

    # Check if required properties are present
    for property in required_properties['float']:
        if property not in properties:
            raise ValueError("Parameter '" +name+ "': property '" + property + "' must be defined")

    # Add default parameters if not present
    for property, default in default_values['float'].iteritems():
        if property not in properties:
            properties[property] = default

    # Finite bounds
    if not np.isfinite(properties['min']) or not np.isfinite(properties['max']):
        raise ValueError("Parameter '" + name + "': 'min' and 'max' must be finite.")
    if (properties['min'] < -1e32 or properties['max'] < -1e32 or properties['min'] < -1e32 or 
        properties['min'] > 1e32  or properties['max'] > 1e32):
        raise ValueError("Parameter '" + name + "': 'min' and 'max' must be finite and between -1e32 and 1e32.")

    # Check compatibility of properties
    if properties['min'] >= properties['max']:
        raise ValueError("Parameter '" + name + "': 'min' should be smaller than 'max'")

    for property, legals in legal_values.iteritems():
        if properties[property] not in legals:
            raise ValueError("Parameter '" +name+ "': invalid value for property '" + property+"'")


@catch_exception
def _validate_enum(name, properties):
    """
    Validate that enum parameter description is valid.

    :param name: Name of parameter
    :type name: str
    :param name: Properties of the parameter
    :type name: dict
    """

    # Check if required properties are present
    for property in required_properties['enum']:
        if property not in properties:
            raise ValueError("Parameter '" +name+ "': property '" + property + "' must be defined")

    # Add default parameters if not present
    for property, default in default_values['enum'].iteritems():
        if property not in properties:
            properties[property] = default

    # Check compatibility of properties
    if len(properties['options']) < 2:
        raise ValueError("Parameter '%s': must give at least 2 options." % name)

    validpat = re.compile('^[a-zA-Z_]+[\w\s]*\Z')
    for option in properties['options']:
        if validpat.match(option) is None:
            raise ValueError("Invalid enum option: %s "
                "Options must be a string beginning with a letter and containing only letters, "
                "numbers, hyphens and underscores ([a-zA-Z0-9_-\s])" % (option))        

    if not all([isinstance(c,python_types['enum']) for c in properties['options']]):
        raise ValueError("Parameter '%s': options must be of type %s." % (name, python_types['enum']))

reformat_from_rest = {'integer': _reformat_integer,
                      'float'  : _reformat_float,
                      'enum'   : _reformat_enum}

validate = {'integer': _validate_integer,
            'float'  : _validate_float,
            'enum'   : _validate_enum}

@catch_exception
def delete_experiment(name='Default name', access_token=None):
    """
    Delete the experiment with the given name.  

    Important, this cancels the experiment and removes all saved results!

    :param name: Experiment name
    :type name: str
    :param access_token: User access token
    :type access_token: str 
   """

    try:
        scientist = Experiment(name=name, resume=True, access_token=access_token)
    except ValueError:
        raise ValueError('Could not delete experiment \''+name+'\' (either it doesn\'t exist or access token is invalid)')
    scientist._client.delete_experiment(scientist.experiment_id)

@catch_exception
def load_config():
    filename = '.whetlab'
    search_path = ['.', os.path.expanduser('~')]
    for dir in search_path:
        full_path = os.path.join(dir, filename)
        if os.path.exists(full_path):
            config = ConfigParser.RawConfigParser()
            config.read(full_path)
            config_dict = {}
            if config.has_option('whetlab', 'access_token'):
                config_dict['access_token'] = config.get('whetlab', 'access_token')
            if config.has_option('whetlab', 'api_url'):
                config_dict['api_url'] = config.get('whetlab', 'api_url')
            return config_dict
    return {}


class Experiment:
    """
    A Whetlab tuning experiment.

    A ``name`` and ``description`` for the experiment must be specified.
    A Whetlab access token must also be provided.
    The parameters to tune in the experiment are specified by
    ``parameters``. It should be a ``dict``, where keys are
    the parameters (``str``) and values are ``dict`` that
    provide information about these parameters. Each of these
    ``dict`` should contain the appropriate keys to properly describe
    the parameter:
    
    * ``'type'``: type of the parameter (default: ``'float'``)
    * ``'min'``: minimum value of the parameter (only for types ``float`` and ``int``)
    * ``'max'``: maximum value of the parameter (only for types ``float`` and ``int``)
    * ``'options'``: list of strings, of the possible values that can take an ``enum`` parameter (only for type ``enum``)
    * ``'size'``: size of parameter (default: ``1``)

    ``outcome`` should also be a ``dict``, describing the outcome. It
    should have the key:

    * ``'name'``: name (``str``) for the outcome being optimized

    If ``name`` match a previously created experiment,
    that experiment will be resumed (in this case, ``parameters`` and ``outcoume`` are ignored).
    This behavior can be avoided by setting the argument ``resume``
    to ``False`` (in which case an error will be raised is an experiment
    with the same name and description is found).    

    :param name: Name of the experiment.
    :type name: str
    :param description: Description of the experiment (default: ``''``).
    :type description: str
    :param parameters: Parameters to be tuned during the experiment (default: ``None``, appropriate when resuming).
    :type parameters: dict
    :param outcome: Description of the outcome to maximize (default: ``None``, appropriate when resuming).
    :type outcome: dict
    :param resume: Whether to allow the resuming of a previously executed experiment. If ``True`` and experiment's name matches an existing experiment, ``parameters`` and ``outcome`` are ignored (default: ``True``).
    :type resume: bool
    :param access_token: Access token for your Whetlab account. If ``None``, then is read from whetlab configuration file (default: ``None``).
    :type access_token: str

    A Whetlab experiment instance will have the following variables:

    :ivar parameters: Parameters to be tuned during the experiment.
    :type parameters: dict
    :ivar outcome: Description of the outcome to maximize.
    :type outcome: dict
    :ivar experiment_id: ID of the experiment (useful for resuming).
    :type experiment_id: int
    """

    @catch_exception
    def __init__(self,
                 name,
                 description='',
                 parameters=None,
                 outcome=None,
                 resume = True,
                 access_token=None):

        # These are for the client to keep track of things without always 
        # querying the REST server ...
        # ... From result IDs to client parameter values
        self._ids_to_param_values = {}
        # ... From result IDs to outcome values
        self._ids_to_outcome_values = {}
        # ... From a parameter name to the setting IDs
        self._param_names_to_setting_ids = {}

        config = load_config()
        if config.has_key('api_url'):
            url = config['api_url']
        else:
            url = DEFAULT_API_URL
        if access_token is None:
            if config.has_key('access_token'):
                access_token = config['access_token']
            else:
                raise Exception("No access token specified in dotfile or via constructor.")

        self._client = SimpleREST(access_token, url)

        # Make a few obvious asserts
        if name == '' or type(name) not in [str,unicode]:
            raise ValueError('Name of experiment must be a non-empty string')

        if type(description)  not in [str,unicode]:
            raise ValueError('Description of experiment must be a string')

        self.experiment = name
        self.experiment_description = description

        self.experiment_id = self._client.find_experiment(self.experiment)

        if self.experiment_id is not None and resume:
            # Sync all the internals with the REST server
            self._sync_with_server()

        else:

            if type(parameters) != dict or len(parameters) == 0:
                raise ValueError('Parameters of experiment must be a non-empty dictionary')
    
            if type(outcome) != dict or len(outcome) == 0:
                raise ValueError('Outcome of experiment must be a non-empty dictionary')

            if 'name' not in outcome:
                raise ValueError('Argument outcome should have key \'name\'')
            self.outcome_name = outcome['name']
            
            # Add specification of parameters to experiment.
            settings = []
            #settings = {}
            for key in parameters.keys():
                param = {}
                param.update(parameters[key])

                for property in param.iterkeys():
                    if property not in supported_properties:
                        raise ValueError("Parameter '" +key+ "': property '" + property + "' not supported")

                # If not type provided, use default
                if not param.has_key('type'):
                    param['type'] = default_values['type']
                ptype = param['type']  
                
                # Map type 'int' to 'integer'
                if cmp(ptype,'int') == 0:
                    ptype = 'integer'
                    param['type'] = 'integer'

                if ptype not in validate:
                    raise ValueError("Parameter '%s' uses unsupported type '%s'." % (key, ptype))
                
                # Check whether description of parameter is valid
                validate[ptype](key,param)

                param['isOutput'] = False
                param['name'] = key
                settings += [param]
                #settings['name'] = param

            # Add the outcome variable
            param = {}
            param.update(outcome)

            # Check outcome doesn't have the same name as any of the parameters
            if outcome['name'] in parameters:
                raise ValueError("Outcome name should not match any of the parameter names")

            # Check if all properties are supported
            for prop in param.iterkeys():
                if prop not in outcome_supported_properties : raise ValueError("Parameter '" +outcome['name']+ "': property '" + prop + "' not supported")
            
            # Check if required properties are present
            for prop in outcome_required_properties:
                if prop not in param : raise ValueError("Parameter '" +key+ "': property '" + prop + "' must be defined")

            # Add default parameters if not present
            for prop, default in outcome_default_values.iteritems():
                if prop not in param: param[prop] = default
            
            # Check compatibility of properties
            for prop, legals in outcome_legal_values.iteritems():
                if param[prop] not in legals : raise ValueError("Parameter '" +key+ "': invalid value for property '" + prop +"'")

            param['isOutput'] = True
            settings += [param]
            #settings[outcome['name']] = param

            # Create experiment.
            try:
                self.experiment_id = self._client.create_experiment(self.experiment, self.experiment_description, settings)
            except Exception as inst:
                # If experiment creation doesn't work, then retry resuming the experiment.
                # This is for cases where two processes are starting an experiment, and
                # one gets to create it first while the other should be resuming it.
                self.experiment_id = self._client.find_experiment(self.experiment)
                if not resume or self.experiment_id is None :
                    raise inst

            # Call _sync_with_server in order to fill-in the state of the object 
            # (e.g. fetching the setting ids)
            self._sync_with_server()

        pending = self.pending()
        if len(pending) > 0:
            print "INFO: this experiment currently has "+str(len(pending))+" jobs (results) that are pending."


    @catch_exception
    def _sync_with_server(self):
        """
        Synchronize the client's internals with the REST server.
        """

        self.experiment, self.experiment_description = self._client.get_experiment_name_and_description(self.experiment_id)

        # Reset internals
        self._ids_to_param_values = {}
        self._ids_to_outcome_values = {}
        self._param_names_to_setting_ids = {}

        # Get settings for this experiment, to get the parameter and outcome names
        rest_parameters = self._client.get_parameters(self.experiment_id)
        self.parameters = {}
        for rest_param in rest_parameters:
            rest_param
            id = rest_param['id']
            name = rest_param['name']
            type = rest_param['type']
            isOutput = rest_param['isOutput']

            self._param_names_to_setting_ids[name] = id

            if isOutput:
                self.outcome_name = name
            else:
                self.parameters[name] = reformat_from_rest[type](rest_param)

        # Get results generated so far for this experiment
        rest_results = self._client.get_results(self.experiment_id)
        # Construct things needed by client internally, to keep track of
        # all the results
        for res in rest_results:
            res_id = res['id']
            variables = res['variables']

            # Construct _ids__param_values dict and ids_to_outcome_values
            self._ids_to_param_values[res_id] = {}
            for v in variables:
                id = v['id']
                name = v['name']
                if cmp(name,self.outcome_name) == 0 :
                    self._ids_to_outcome_values[res_id] = v['value']
                else:
                    self._ids_to_param_values[res_id][v['name']] = v['value']

    @catch_exception
    def suggest(self):
        """
        Suggest a new job.

        :return: Values to assign to the parameters in the suggested job.
        :rtype: dict
        """

        result_id = self._client.get_suggestion(self.experiment_id)

        # Poll the server for the actual variable values in the suggestion.  
        result = self._client.get_result(result_id)
        variables = result['variables']
        while not variables:
            time.sleep(2)
            result = self._client.get_result(result_id)
            variables = result['variables']

        # Put in nicer format
        next = {}
        for var in variables:
            # Don't return the outcome variable
            if cmp(var['name'],self.outcome_name) != 0:
                if isinstance(var['value'], str):
                    value = ast.literal_eval(var['value'])
                else:
                    value = var['value']

                if isinstance(value, list) or isinstance(value, np.ndarray):
                    next[var['name']] = [python_types[self.parameters[var['name']]['type']](v) for v in value]
                else:
                    next[var['name']] = python_types[self.parameters[var['name']]['type']](value)
            
        # Keep track of id / param_values relationship
        self._ids_to_param_values[result_id] = next
        next = Result(next)
        next._result_id = result_id
        next._experiment_id = self.experiment_id

        return next

    @catch_exception
    def get_by_result_id(self, id):
        """
        Return the parameter values corresponding to the given unique job/result ``id``.
        If no result matches, return ``None``.

        :param id: Unique result identifier
        :type id: int       
        :return: ID of the corresponding result. If not match, None is returned.
        :rtype: dict or None
        """
        result = None
        if id in self._ids_to_param_values:            
            result = Result(self._ids_to_param_values[id])

        else:
            self._sync_with_server()
            if id in self._ids_to_param_values:
                result = Result(self._ids_to_param_values[id])

        if result is not None:
            result._result_id = id
            result._experiment_id = self.experiment_id

        return result

    @catch_exception
    def get_id(self, param_values):
        """
        Return the result ID corresponding to the given ``param_values``.
        If no result matches, return ``None``.

        :param param_values: Values of parameters.
        :type param_values: dict       
        :return: ID of the corresponding result. If not match, None is returned.
        :rtype: int or None
        """

        if type(param_values) == 'whetlab.Result' and param_values._result_id is not None:
            if param_values._experiment_id == self.experiment_id:
                return param_values._result_id
            else: # Result belongs to another experiment
                param_values = dict(param_values)

        # Sync with the REST server
        self._sync_with_server()

        id = None

        for k,v in self._ids_to_param_values.iteritems():
            if cmp(v,param_values) == 0:
                id = k

        return id

    @catch_exception
    def get_all_results(self):
        """
        Return a list of all jobs and a list of their corresponding outcomes.
        Pending outcomes are returned as having ``None`` outcomes.

        :return: Tuple of lists containing parameter values and corresponding outcomes indexed by unique result id.
        :rtype: tuple of lists
        """

        # Sync with the REST server
        self._sync_with_server()

        jobs     = []
        outcomes = []
        for k,v in self._ids_to_param_values.iteritems():
            if v:
                jobs.append(Result(v))
                jobs[-1]._result_id = k
                jobs[-1]._experiment_id = self.experiment_id
                outcomes.append(self._ids_to_outcome_values.get(k, None))            

        return jobs, outcomes

    @catch_exception
    def cancel_by_result_id(self, id):
        """
        Delete the experiment indexed by the given unique job/result ``id``.

        :param id: Unique result identifier
        :type id: int
        """

        self._client.delete_result(id)

    @catch_exception
    def update_by_result_id(self, result_id, outcome_val):
        """
        Update the experiment with the outcome value indexed by the given unique job/result ``id``.

        :param result_id: Unique result identifier
        :type id: int
        :param outcome_val: Outcome value associated with this result
        :type outcome_val: float
        """

        if outcome_val is not None:
            outcome_val = float(outcome_val)

        result_id = int(result_id)

        result = self._client.get_result(result_id)
        if result is None or 'variables' not in result:
            raise ValueError("Job with result_id '" + str(result_id) + "' not found.")

        for var in result['variables']:
            if var['name'] == self.outcome_name:
                var['value'] = outcome_val
                break # Assume only one outcome per experiment!
        self._client.update_result(result_id,result)
        self._ids_to_outcome_values[result_id] = outcome_val

    @catch_exception
    def update(self, param_values, outcome_val):
        """
        Update the experiment with the outcome value associated with some parameter values.

        :param param_values: Values of parameters.
        :type param_values: dict
        :param outcome_val: Value of the outcome.
        :type outcome_val: type defined for outcome
        """
        if outcome_val is not None:
            outcome_val = float(outcome_val)
        # Check if param_values is compatible
        for param,value in param_values.iteritems():
            if param not in self.parameters:
                raise ValueError("Parameter '" +param+ "' not valid")

            if np.array(value).size != self.parameters[param]['size']:
                raise ValueError("Parameter '" +param+ "' should be of size "+str(self.parameters[param]['size']))
            
            if self.parameters[param]['type'] == 'float' or self.parameters[param]['type'] == 'integer':
                if ((not np.all(np.isfinite(value))) or
                     np.any(np.array(value) < self.parameters[param]['min']) or
                     np.any(np.array(value) > self.parameters[param]['max'])):
                    raise ValueError("Parameter '" +param+ "' should have value between "+str(self.parameters[param]['min']) +" and " + str(self.parameters[param]['max']))
            
            if self.parameters[param]['type'] == 'enum':
                list_value = [value] if type(value) == str else value
                if not np.all([ v in self.parameters[param]['options'] for v in list_value]):
                    raise ValueError("Enum parameter '" +param+ "' should take values in " + str(self.parameters[param]['options']))
            
            if isinstance(value, np.ndarray) or isinstance(value, list):
                value_type = {type(np.asscalar(np.array(v))) for v in value}
                if len(value_type) > 1:
                    raise TypeError('All returned values for a variable should have the same type.')
                value_type = value_type.pop()
            else:
                value_type = type(value)
            if value_type != python_types[self.parameters[param]['type']]:
                raise TypeError("Parameter '" +param+ "' should be of type " + self.parameters[param]['type'])

        # Check is all parameter values are specified
        for param in self.parameters.keys():
            if param not in param_values:
                raise ValueError("Parameter '" +param+ "' not specified")

        # Check whether this param_values has a results ID
        if (type(param_values) == Result and 
                param_values._result_id is not None and
                param_values._experiment_id == self.experiment_id):        
            result_id = param_values._result_id
        else:
            result_id = self.get_id(param_values)

        if result_id is None:
            # If not, then this is a result that was not suggested,
            # must add it to the server

            # Create variables for new result
            variables = []
            for name, setting_id in self._param_names_to_setting_ids.iteritems():
                if name in param_values:
                    value = param_values[name]
                elif name == self.outcome_name:
                    if outcome_val is None:
                        continue # We don't put in None, simply leave empty
                    value = outcome_val
                else:
                    raise ValueError('Failed to update with non-suggested experiment')
                variables += [{'setting':setting_id, 'result':result_id, 
                           'name':name, 'value':value}]


            result_id = self._client.add_result(variables, self.experiment_id, self.experiment_description)

            self._ids_to_param_values[result_id] = param_values
            self._ids_to_outcome_values[result_id] = outcome_val
            
        else:
            # Fill in result with the given outcome value
            if outcome_val is not None:
                result = self._client.get_result(result_id)
                for var in result['variables']:
                    if var['name'] == self.outcome_name:
                        var['value'] = outcome_val
                        break # Assume only one outcome per experiment!
                self._client.update_result(result_id,result)
                self._ids_to_outcome_values[result_id] = outcome_val

    @catch_exception
    def cancel(self,param_values):
        """
        Cancel a job, by removing it from the jobs recorded so far in the experiment.

        :param param_values: Values of the parameters for the job to cancel.
        :type param_values: dict
        """

        # Check whether this param_values has a results ID
        id = self.get_id(param_values)

        if id is not None:
            # Delete from internals
            del self._ids_to_param_values[id]
            if id in self._ids_to_outcome_values:
                del self._ids_to_outcome_values[id]

            # Delete from server
            self._client.delete_result(id)
        else:
            print 'Did not find experiment with the provided parameters'

    @catch_exception
    def pending(self):
        """
        Return the list of jobs which have been suggested, but for which no 
        result has been provided yet.

        :return: List of parameter values.
        :rtype: list
        """
    
        # Sync with the REST server     
        self._sync_with_server()
        
        # Find IDs of results with value None and append parameters to returned list
        ret = [] 
        for key,val in self._ids_to_outcome_values.iteritems():
            if val is None:
                ret.append(Result(self._ids_to_param_values[key]))
                ret[-1].result_id = key
                ret[-1].experiment_id = self.experiment_id

        return list(ret)

    @catch_exception
    def clear_pending(self):
        """
        Cancel jobs (results) that are marked as pending.
        """

        p = self.pending()
        for job in p:
            self.cancel(job)

    @catch_exception
    def best(self):
        """
        Return job with best outcome found so far.
        
        :return: Parameter values with best outcome.
        :rtype: dict
        """

        # Sync with the REST server
        self._sync_with_server()

        # Find ID of result with best outcome
        ids = np.array(self._ids_to_outcome_values.keys())
        outcomes = [self._ids_to_outcome_values[i] for i in ids]

        # Clean up nans, infs and Nones
        outcomes = np.array(map(lambda x: float(x) if x is not None else -np.inf, outcomes))
        outcomes[np.logical_not(np.isfinite(outcomes))] = -np.inf
        result_id = ids[outcomes.argmax()]

        result = Result(self._ids_to_param_values[result_id])
        result._result_id     = result_id
        result._experiment_id = self.experiment_id

        return result
    
    @catch_exception
    def report(self):
        """
        Plot a visual report of the progress made so far in the experiment.
        """

        # Sync with the REST server
        self._sync_with_server()

        # Report historical progress and results assumed pending
        import matplotlib.pyplot as plt        

        # Get outcome values and put them in order of their IDs,
        # which should be equivalent to chronological order (of suggestion time)
        ids = np.array(self._ids_to_outcome_values.keys())
        outcomes_values = np.array(self._ids_to_outcome_values.values())

        # Clean up nans, infs and Nones
        outcomes_values = np.array(map(lambda x: float(x) if x is not None else -np.inf, outcomes_values))
        outcomes_values[np.logical_not(np.isfinite(outcomes_values))] = -np.inf

        s = ids.argsort()
        ids = ids[s]
        outcome_values = outcomes_values[s]
        outcome_values = np.array([float(i) for i in outcome_values])
        if outcome_values.size == 0 or np.all(np.isinf(outcome_values)):
            print 'There are no completed results to report'
            return

        # Plot progression
        plt.figure(1)
        plt.clf()
        y = outcome_values
        best_so_far = [ np.max(y[:(i+1)]) for i in range(len(y)) ]
        plt.scatter(range(len(y)),y,marker='x',color='k',label='Outcomes')
        plt.plot(range(len(y)),best_so_far,color='k',label='Best so far')
        plt.xlabel('Result #')
        plt.ylabel(self.outcome_name)
        plt.title('Results progression')
        plt.legend(loc=3)
        plt.draw()
        plt.ion()
        plt.show()
        
        # Plot table of results
        plt.figure(2)
        param_names = list(np.sort(self.parameters.keys()))
        col_names = ['Result #'] + param_names + [self.outcome_name]
        cell_text = []
        for nb,id in enumerate(ids):
            # Get paramater values, put in correct order and add to
            # table with corresponding outcome value
            params, values = zip(*self._ids_to_param_values[id].iteritems())
            s = np.argsort(params)
            values = np.array(values)[s]
            outcome = self._ids_to_outcome_values[id]
            cell_text.append([str(nb+1)] + [str(v) for v in values] + [str(outcome)])

        if len(cell_text) > 20:
            cell_text = cell_text[-20:]
        the_table = plt.table(cellText = cell_text, colLabels=col_names, loc='center')

        ## change cell properties
        table_props=the_table.properties()
        table_cells=table_props['child_artists']
        for cell in table_cells:
            cell.set_fontsize(8)

        plt.axis('off')
        plt.title('Table of results')
        plt.draw()
        plt.ion()
        plt.show()



class Result(dict):
    """
    Simple class for results, which contain a result ID as metadata.
    """

    _experiment_id = None
    _result_id     = None    


RETRY_TIMES = [5,30,60,150,300]

def retry(f):
    @functools.wraps(f)
    def func(*args, **kwargs):
        for i in range(len(RETRY_TIMES)+1):
            try:
                return f(*args, **kwargs)
            except requests.exceptions.ConnectionError as e:
                if i == len(RETRY_TIMES):
                    raise e
                if i >=3 : # Only warn starting at the 2nd retry
                    print 'WARNING: experiencing problems communicating with the server. Will try again in ',RETRY_TIMES[i],' seconds.'
                time.sleep(RETRY_TIMES[i])
            except ClientError as e: # An explicit error was returned by the server
                if e.code == 503: # Temporary server maintenance
                    retry_time = np.round(np.random.rand()*2*30)
                    i -= 1
                    print 'WARNING: Server is undergoing temporary maintenance. Will try again in %d seconds.' % (retry_time)
                    time.sleep(retry_time)
                elif e.code == 429:
                    if i == len(RETRY_TIMES):
                        i -= 1
                    msg = e.message[0] if type(e.message) == list else e.message
                    print 'WARNING: rate limited by the server: %s Will try again in %d seconds.' % (msg, RETRY_TIMES[i])
                    time.sleep(RETRY_TIMES[i])
                elif e.code > 500:
                    if i == len(RETRY_TIMES):
                        raise e
                    if i >=3 : # Only warn starting at the 2nd retry
                        print 'WARNING: experiencing problems communicating with the server. Will try again in ',RETRY_TIMES[i],' seconds.'
                    time.sleep(RETRY_TIMES[i])
                else:
                    raise
            except:
                raise 
                       
    return func

class SimpleREST:
    """
    Simple class that wraps calls to the web server through the REST API.

    The main reason for this class is to deal with retries, to be
    robust to glitches in the communication with the server.
    """

    def __init__(self, access_token, url):
        

        # Create REST server client
        options = ({'headers' : {'Authorization':'Bearer ' + access_token}, 
                    'user_agent':'whetlab_python_client',
                    'api_version':'api',
                    'base': url})
        
        self._client = server.Client({},options)

    @retry
    def create_experiment(self, name, description, settings):
        """
        Create experiment and return its ID.

        :param name: Name of experiment
        :type name: str
        :param description: Description of experiment
        :type description: str
        :param settings: Specification of the experiment's variables
        :type settings: list
        :return: Experiment ID
        :rtype: int
        """
        
        res = self._client.experiments().create(name=name, 
                                                description=description,
                                                settings=settings)
        return res.body['id']

    @retry
    def delete_experiment(self, id):
        """
        Delete experiment with the given ID ``id``.

        :param id: ID of experiment to delete
        :type ind: int
        """

        res = self._client.experiment(str(id)).delete()
        print 'Experiment has been deleted'

    @retry
    def find_experiment(self, name):
        """
        Look for experiment matching name and return its ID.

        :param name: Experiment's name
        :type name: str
        :return: Experiment's ID.
        :rtype: int
        """

        # Search one page at a time
        page = 1
        more_pages = True
        while more_pages:
            rest_exps = self._client.experiments().get({'query':{'page':page}}).body

            # Check if more pages to come
            more_pages = rest_exps['next'] is not None
            page += 1
        
            # Find in current page whether we find the experiment we are looking for
            rest_exps = rest_exps['results']
            for exp in rest_exps:
                if cmp(exp['name'],name) == 0:
                    return exp['id']
        return None

    @retry
    def get_experiment_name_and_description(self, id):
        """
        Gives the name and description of an experiment, from it's ID

        :param id: Experiment's ID
        :type id: int
        :return: Name and description of experiment
        :rtype: tuple (pair of str)
        """
        res = self._client.experiments().get({'query':{'id':id}}).body['results'][0]
        return res['name'], res['description']
        
    @retry
    def get_parameters(self, id):
        """
        Gives the parameters of an experiment, from it's ID

        :param id: Experiment's ID
        :type id: int
        :return: Parameters of the experiment
        :rtype: dict
        """

        return self._client.settings().get(str(id),{'query':{'page_size':INF_PAGE_SIZE}}).body['results']

    @retry
    def get_results(self, id):
        """
        Gives the results of an experiment, from it's ID

        :param id: Experiment's ID
        :type id: int
        :return: Results of the experiment
        :rtype: list
        """

        return self._client.results().get({'query': {'experiment':id,'page_size':INF_PAGE_SIZE}}).body['results']

    @retry
    def get_suggestion(self, id):
        """
        Get suggestion. Obtained in the form of a result ID.

        :param id: Experiment's ID 
        :type id: int
        :return: Suggested result's ID
        :rtype: int
        """

        return  self._client.suggest(str(id)).go().body['id']

    @retry
    def get_result(self, result_id):
        """
        Get a result from its ID.

        :param id: Result's ID 
        :type id: int
        :return: Description of the result
        :rtype: dict
        """
        return self._client.result(str(result_id)).get().body

    @retry
    def add_result(self, variables, id, experiment_description):
        """
        Add a result with variable assignments from ``variables``,
        to experiment with ID ``id``.

        :param variables: Parameter and outcome values for a result
        :type variables: dict
        :param id: experiment's ID 
        :type id: int
        :param experiment_description: Description of experiment
        :type experiment_description: str
        :return: Result ID of the added result
        :rtype: int
        """
        return self._client.results().add(variables, id, True, experiment_description).body['id']

    @retry
    def update_result(self, result_id, result):
        """
        Update a result from its ID ``result_id``, based on the content of ``result``.

        :param result_id: ID of the result
        :type result_id: int
        :param result: Parameter and outcome values of the result
        :type result: dict
        """
        self._client.result(str(result_id)).update(**result)
                        
    @retry
    def delete_result(self, result_id):
        """
        Delete a result from its ID ``result_id``.

        :param result_id: ID of the result
        :type result_id: int
        """
        self._client.result(str(result_id)).delete()

