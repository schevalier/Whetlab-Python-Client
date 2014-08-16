import os
import ast
import tempfile
import ConfigParser
import collections
import numpy as np
import server
import time
import functools
import requests

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

DEFAULT_API_URL = 'http://whetlab-server.elasticbeanstalk.com'

supported_properties = set(['min','max','size','scale','units','type','options'])
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

outcome_supported_properties = set(['units','type','name'])
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

    # Check compatibility of properties
    if properties['min'] >= properties['max']:
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
    if len(properties['options']) < 3:
        raise ValueError("Parameter '%s': must give at least 3 options." % name)

    if not all([isinstance(c,python_types['enum']) for c in properties['options']]):
        raise ValueError("Parameter '%s': options must be of type %s." % name, python_types['enum'])

reformat_from_rest = {'integer': _reformat_integer,
                      'float'  : _reformat_float,
                      'enum'   : _reformat_enum}

validate = {'integer': _validate_integer,
            'float'  : _validate_float,
            'enum'   : _validate_enum}


@catch_exception
def delete_experiment(access_token, name):
    """
    Delete the experiment with the given name.  

    Important, this cancels the experiment and removes all saved results!

    :param access_token: User access token
    :type access_token: str
    :param name: Experiment name
    :type name: str
    """

    try:
        scientist = Experiment(access_token, name, resume=True)
    except ValueError:
        raise ValueError('Could not delete experiment \''+name+'\' (either it doesn\'t exist or access token is invalid)')
    scientist._delete()

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
    
    * ``'min'``: minimum value of the parameter
    * ``'max'``: maximum value of the parameter
    * ``'scale'``: scale to use when exploring parameter values (default: ``'linear'``)
    * ``'units'``: units (``str``) in which the parameter is measured (default: ``''``)
    * ``'type'``: type of the parameter (default: ``'float'``)
    * ``'size'``: size of parameter (default: ``1``)

    ``outcome`` should also be a ``dict``, describing the outcome. It
    should have the keys:

    * ``'name'``: name (``str``) for the outcome being optimized
    * ``'type'``: type of the parameter, either ``'float'``, ``'int'`` or  ``'enum'`` (default: ``'float'``)
    * ``'units'``: units (``str``) in which the parameter is measured (default: ``''``)

    If ``name`` and ``description`` match a previously created experiment,
    that experiment will be resumed (in this case, ``parameters`` and ``outcoume`` are ignored).
    This behavior can be avoided by setting the argument ``resume``
    to ``False`` (in which case an error will be raised is an experiment
    with the same name and description is found).    

    :param access_token: Access token for your Whetlab account.
    :type access_token: str
    :param name: Name of the experiment.
    :type name: str
    :param description: Description of the experiment.
    :type description: str
    :param parameters: Parameters to be tuned during the experiment.
    :type parameters: dict
    :param outcome: Description of the outcome to maximize.
    :type outcome: dict
    :param resume: Whether to allow the resuming of a previously executed experiment.
    :type resume: bool

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
                 access_token=None,
                 name='Default name',
                 description='Default description',
                 parameters=None,
                 outcome=None,
                 resume = True,
                 url=None):

        # These are for the client to keep track of things without always 
        # querying the REST server ...
        # ... From result IDs to client parameter values
        self._ids_to_param_values = {}
        # ... From result IDs to outcome values
        self._ids_to_outcome_values = {}
        # ... From a parameter name to the setting IDs
        self._param_names_to_setting_ids = {}

        config = load_config()
        if url is None:
            if config.has_key('api_url'):
                url = config['api_url']
            else:
                url = DEFAULT_API_URL
        if access_token is None:
            if config.has_key('access_token'):
                access_token = config['access_token']
            else:
                raise Exception("No access token specified in dotfile or via constructor.")

        # Create REST server client
        options = ({'headers' : {'Authorization':'Bearer ' + access_token}, 
                    'user_agent':'whetlab_python_client',
                    'api_version':'api',
                    'base': url})
        
        self._client = server.Client({},options)

        # Make a few obvious asserts
        if name == '' or type(name) not in [str,unicode]:
            raise ValueError('Name of experiment must be a non-empty string')

        if type(description)  not in [str,unicode]:
            raise ValueError('Description of experiment must be a string')

        self.experiment = name
        self.experiment_description = description

        self.experiment_id = self._find_experiment(self.experiment)

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

                ptype = param['type'] if param.has_key('type') else default_values['type']

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
            for property in param.iterkeys():
                if property not in outcome_supported_properties : raise ValueError("Parameter '" +key+ "': property '" + property + "' not supported")
            
            # Check if required properties are present
            for property in outcome_required_properties:
                if property not in param : raise ValueError("Parameter '" +key+ "': property '" + property + "' must be defined")

            # Add default parameters if not present
            for property, default in outcome_default_values.iteritems():
                if property not in param: param[property] = default
            
            # Check compatibility of properties
            for property, legals in outcome_legal_values.iteritems():
                if param[property] not in legals : raise ValueError("Parameter '" +key+ "': invalid value for property '" + property+"'")

            param['isOutput'] = True
            settings += [param]
            #settings[outcome['name']] = param

            # Create experiment.
            try:
                res = self._client.experiments().create(name=self.experiment, 
                                                        description=self.experiment_description,
                                                        settings=settings)
                self.experiment_id = res.body['id']                
            except Exception as inst:
                # If experiment creation doesn't work, then retry resuming the experiment.
                # This is for cases where two processes are starting an experiment, and
                # one gets to create it first while the other should be resuming it.
                self.experiment_id = self._find_experiment(self.experiment)
                if not resume or self.experiment_id is None :
                    raise inst

            # Call _sync_with_server in order to fill-in the state of the object 
            # (e.g. fetching the setting ids)
            self._sync_with_server()

        pending = self.pending()
        if len(pending) > 0:
            print "INFO: this experiment currently has "+str(len(pending))+" jobs (results) that are pending."

    def _find_experiment(self, name):
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

    @catch_exception
    def _sync_with_server(self):
        """
        Synchronize the client's internals with the REST server.
        """

        res = self._client.experiments().get({'query':{'id':self.experiment_id}}).body['results'][0]
        self.experiment = res['name']
        self.experiment_description = res['description']

        # Reset internals
        self._ids_to_param_values = {}
        self._ids_to_outcome_values = {}
        self._param_names_to_setting_ids = {}

        # Get settings for this experiment, to get the parameter and outcome names
        rest_parameters = self._client.settings().get(str(self.experiment_id),{'query':{'page_size':INF_PAGE_SIZE}}).body
        rest_parameters = rest_parameters['results']

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
        rest_results = self._client.results().get({'query': {'experiment':self.experiment_id,'page_size':INF_PAGE_SIZE}}).body['results']
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

        res = self._client.suggest(str(self.experiment_id)).go()
        result_id = res.body['id']

        # Poll the server for the actual variable values in the suggestion.  
        variables = res.body['variables']
        while not variables:
            time.sleep(2)
            result = self._client.result(str(result_id)).get()
            variables = result.body['variables']

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
        return next


    @catch_exception
    def _get_id(self,param_values):
        """
        Return the result ID corresponding to the given ``param_values``.
        If no result matches, return ``None``.

        :param param_values: Values of parameters.
        :type param_values: dict       
        :return: ID of the corresponding result. If not match, None is returned.
        :rtype: int or None
        """

        # Sync with the REST server
        self._sync_with_server()

        id = None

        for k,v in self._ids_to_param_values.iteritems():
            if cmp(v,param_values) == 0:
                id = k

        return id
    

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

            if self.parameters[param]['type'] == 'float' or self.parameters[param]['type'] == 'integer':
                if np.any(np.array(value) < self.parameters[param]['min']) or np.any(np.array(value) > self.parameters[param]['max']):
                    raise ValueError("Parameter '" +param+ "' should have value between "+str(self.parameters[param]['min']) +" and " + str(self.parameters[param]['max']))
            
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
        result_id = self._get_id(param_values)

        if result_id is None:
            # If not, then this is a result that was not suggested,
            # must add it to the server

            ## Get a time stamp for this submitted result
            #import datetime
            #import json
            #
            #dthandler = lambda obj: (
            #   obj.isoformat()
            #   if isinstance(obj, datetime.datetime)
            #   or isinstance(obj, datetime.date)
            #   else None)
            #date = json.loads(json.dumps(datetime.datetime.now(), default=dthandler))            

            # Create variables for new result
            variables = []
            for name, setting_id in self._param_names_to_setting_ids.iteritems():
                if name in param_values:
                    value = param_values[name]
                elif name == self.outcome_name:
                    value = outcome_val
                else:
                    raise ValueError('Failed to update with non-suggested experiment')
                variables += [{'setting':setting_id, 'result':result_id, 
                           'name':name, 'value':value}]

            res = self._client.results().add(variables, self.experiment_id, True, self.experiment_description)
            result_id = res.body['id']

            self._ids_to_param_values[result_id] = param_values
            
        else:
            # Fill in result with the given outcome value
            result = self._client.result(str(result_id)).get().body
            for var in result['variables']:
                if var['name'] == self.outcome_name:
                    var['value'] = outcome_val
                    self._ids_to_outcome_values[result_id] = var
                    break # Assume only one outcome per experiment!
            res = self._client.result(str(result_id)).update(**result)
            self._ids_to_outcome_values[result_id] = outcome_val

    @catch_exception
    def cancel(self,param_values):
        """
        Cancel a job, by removing it from the jobs recorded so far in the experiment.

        :param param_values: Values of the parameters for the job to cancel.
        :type param_values: dict
        """

        # Check whether this param_values has a results ID
        id = self._get_id(param_values)

        if id is not None:
            # Delete from internals
            del self._ids_to_param_values[id]
            if id in self._ids_to_outcome_values:
                del self._ids_to_outcome_values[id]

            # Delete from server
            self._client.result(str(id)).delete()
        else:
            print 'Did not find experiment with the provided parameters'


    @catch_exception
    def _delete(self):
        """
        Delete the experiment with the given name and description.  
        
        Important, this cancels the experiment and removes all saved results!
        
        """

        res = self._client.experiment(str(self.experiment_id)).delete()
        print 'Experiment has been deleted'

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
                ret.append(self._ids_to_param_values[key])
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
        # Change Nones with infs
        outcomes = np.array(map(lambda x: x if x is not None else np.inf, outcomes))
        result_id = ids[outcomes.argmax()]
        return self._ids_to_param_values[result_id]
    
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

        # Change Nones with infs
        outcomes_values = np.array(map(lambda x: x if x is not None else np.inf, outcomes_values))
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



