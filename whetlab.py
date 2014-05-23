import tempfile
import collections
import numpy as np
import whetlab_api
import time

INF_PAGE_SIZE = 1000000

supported_properties = set(['min','max','size','scale','units','type'])
required_properties = set(['min','max'])
default_values = {'size':1,
          'scale':'linear',
          'units':'',
          'type':'float'}
legal_values = {'size':set([1]),
        'scale':set(['linear','log']),
        'type':set(['float', 'integer'])}

outcome_supported_properties = set(['units','type','name'])
outcome_required_properties = set(['name'])
outcome_default_values = {'min':-10.,
              'max':10.,
              'size':1,
              'scale':'linear',
              'units':'',
              'type':'float'}
outcome_legal_values = {'size':set([1]),
            'scale':set(['linear']),
            'type':set(['float'])}

class Experiment:
    """
    A Whetlab tuning experiment.

    A name and description for the experiment must be specified.
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

    Outcome should also be a ``dict``, describing the outcome. It
    should have the keys:

    * ``'name'``: name (``str``) for the outcome being optimized
    * ``'type'``: type of the parameter (default: ``'float'``)
    * ``'units'``: units (``str``) in which the parameter is measured (default: ``''``)

    Finally, experiments can be resumed from a previous state.
    To do so, ``name`` must match a previously created experiment
    and argument ``resume`` must be set to ``True`` (default is ``False``).

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
    :param resume: Whether to resume a previously executed experiment. If True, ``parameters`` and ``outcome`` are ignored.
    :type resume: bool
    :param experiment_id: ID of the experiment to resume. If not specified, will try to match experiment based on the provided name and description.
    :type experiment_id: int
    :param task_id: ID of the task to resume. If not specified, will try to match task based on the provided name.
    :type task_id: int

    A Whetlab experiment instance will have the following variables:

    :ivar parameters: Parameters to be tuned during the experiment.
    :type parameters: dict
    :ivar outcome: Description of the outcome to maximize.
    :type outcome: dict
    :ivar experiment_id: ID of the experiment (useful for resuming).
    :type experiment_id: int
    :ivar task_id: ID of the task (useful for resuming).
    :type task: id
    """

    def __init__(self,
             access_token,
             name='Default name',
             description='Default description',
             parameters=None,
             outcome=None,
             resume = False):

        # These are for the client to keep track of things without always 
        # querying the REST server ...
        # ... From result IDs to client parameter values
        self._ids_to_param_values = {}
        # ... From result IDs to outcome values
        self._ids_to_outcome_values = {}
        # ... From a parameter name to the setting IDs
        self._param_names_to_setting_ids = {}
        # ... The set of result IDs corresponding to suggested jobs that are pending
        #     *in this current instance of the client*
        self._pending = set([])

        # Create REST server client
        options = ({'headers' : {'Authorization':'Bearer ' + access_token}, 
                    'user_agent':'whetlab_python_client',
                    'api_version':'api',
                    'base': 'http://api.whetlab.com'})
        
        self._client = whetlab_api.Client({},options)

        # Make a few obvious asserts
        if name == '' or type(name) != str:
            raise ValueError('Name of experiment must be a non-empty string')

        if type(description) != str:
            raise ValueError('Description of experiment must be a string')

        if type(parameters) != dict or len(parameters) == 0:
            raise ValueError('Parameters of experiment must be a non-empty dictionary')

        if type(outcome) != dict or len(outcome) == 0:
            raise ValueError('Outcome of experiment must be a non-empty dictionary')

        # For now, we support one task per experiment, and the name and description of the task
        # is the same as the experiment's
        self.experiment = name
        self.experiment_description = description
        self.task = name
        self.task_description = description

        if resume:
            # Sync all the internals with the REST server
            self.experiment_id = None
            self.task_id = None
            self._sync_with_server()
        else:
            if parameters is None or outcome is None:
                raise ValueError("Parameters and outcome must be specified")
            
            # Create new experiment
            res = self._client.experiments().create(name=self.experiment,description=self.experiment_description,user=4)
            experiment_id = res.body['id']
            self.experiment_id = experiment_id

            # Create a task for this experiment
            from datetime import datetime
            task_name = str(datetime.now())
            try:
                res = self._client.tasks().create(experiment=experiment_id,name=self.task,description=self.task_description)
            except:
                # Need to try to clean up the experiment if task creation failed
                res = self._client.experiment(experiment_id).delete()
                raise
            self.task_id = res.body['id']

            if 'name' not in outcome:
                raise ValueError('Argument outcome should have key \'name\'')
            self.outcome_name = outcome['name']
            
            # Add specification of parameters to task
            self.parameters = {}
            for key in parameters.keys():
                param = {}
                param.update(parameters[key])

                # Check if all properties are supported
                if param['type'] is 'enum':
                    raise ValueError("Enum types are not supported yet.  Please use integers instead.")

                for property in param.iterkeys():
                    if property not in supported_properties : raise ValueError("Parameter '" +key+ "': property '" + property + "' not supported")
                
                # Check if required properties are present
                for property in required_properties:
                    if property not in param : raise ValueError("Parameter '" +key+ "': property '" + property + "' must be defined")

                # Add default parameters if not present
                for property, default in default_values.iteritems():
                    if property not in param: param[property] = default
                
                # Check compatibility of properties
                if param['min'] >= param['max'] : raise ValueError("Parameter '" + key + "': 'min' should be smaller than 'max'")
                for property, legals in legal_values.iteritems():
                    if param[property] not in legals : raise ValueError("Parameter '" +key+ "': invalid value for property '" + property+"'")

                self.parameters[key] = param
                #res = self._client.setting().set(name=key,type=param['type'],min=param['min'],max=param['max'],size=param['size'],
                #                units=param['units'],experiment=experiment_id, scale=param['scale'], isOutput=False)
                res = self._client.setting().set(name=key,experiment=experiment_id, isOutput=False, **param)
                self._param_names_to_setting_ids[key] = res.body['id']

            # Add the outcome variable
            param = {}
            param.update(outcome)

            # Check outcome doesn't have the same name as any of the parameters
            if outcome['name'] in self.parameters:
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

            res = self._client.setting().set(experiment=experiment_id, isOutput=True, **param)
            self._param_names_to_setting_ids[outcome['name']] = res.body['id']


    def _sync_with_server(self):
        """
        Synchronize the client's internals with the REST server.
        """

        # Reset internals
        self._ids_to_param_values = {}
        self._ids_to_outcome_values = {}
        self._param_names_to_setting_ids = {}

        found = False

        if self.experiment_id is None:
            # Look for experiment and get the ID... search one page at a time
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
                    if cmp(exp['description'],self.experiment_description) == 0 \
                            and cmp(exp['name'],self.experiment) == 0:
                        self.experiment_id = exp['id']
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError('Experiment with name \''+self.experiment+'\' and description \''+self.experiment_description+'\' not found')
        else:
            res = self._client.experiments().get({'query':{'id':self.experiment_id}}).body['results'][0]
            self.experiment = res['name']
            self.experiment_description = res['description']
            

        if self.task_id is None:
            page = 1
            more_pages = True
            while more_pages:
                rest_tasks = self._client.tasks().get({'query':{'page':page}}).body
            
                # Check if more pages to come
                more_pages = rest_tasks['next'] is not None
                page += 1

                # Find in current page whether we find the task we are looking for
                rest_tasks = rest_tasks['results']
                found = False
                for task in rest_tasks:
                    if cmp(task['experiment'],self.experiment_id) == 0\
                            and cmp(task['name'],self.task) == 0\
                            and cmp(task['description'],self.task_description) == 0:
                        self.task_id = task['id']
                        found = True
                        break
            if not found:
                raise ValueError('Task with name \''+self.task+'\' and description \''+self.task_description+'\' not found')
        else:
            res = self._client.tasks().get({'query':{'id':self.task_id}}).body['results'][0]
            self.task = res['name']
            self.task_description = res['description']
            

        # Get settings for this task, to get the parameter and outcome names
        rest_parameters = self._client.settings().get(str(self.experiment_id),{'query':{'page_size':INF_PAGE_SIZE}}).body
        rest_parameters = rest_parameters['results']
        self.parameters = {}
        for param in rest_parameters:
            id = param['id']
            name = param['name']
            type=param['type']
            min=param['min']
            max=param['max']
            size=param['size']
            units=param['units']
            scale=param['scale']
            isOutput=param['isOutput']

            self._param_names_to_setting_ids[name] = id

            if isOutput:
                self.outcome_name = name
            else:
                self.parameters[name] = {'type':type,'min':min,'max':max,
                             'size':size,'units':units,'scale':scale}

        # Get results generated so far for this task
        rest_results = self._client.results().get({'query': {'task':self.task_id,'page_size':INF_PAGE_SIZE}}).body['results']
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


    def __del__(self):
        """
        Remove the suggested experiments that have not been updated.
        """

        for id in list(self._pending):
            self._client.result(str(id)).delete()

    def suggest(self):
        """
        Suggest a new job.

        :return: Values to assign to the parameters in the suggested job.
        :rtype: dict
        """

        res = self._client.suggest(str(self.task_id)).go()
        result_id = res.body['id']
        # Remember that this job is now assumed to be pending
        self._pending.add(result_id)

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
                next[var['name']] = var['value']
            
        # Keep track of id / param_values relationship
        self._ids_to_param_values[result_id] = next
        return next

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
    
    def update(self, param_values, outcome_val):
        """
        Update the experiment with the outcome value associated with some parameter values.

        :param param_values: Values of parameters.
        :type param_values: dict
        :param outcome_val: Value of the outcome.
        :type outcome_val: type defined for outcome
        """

        outcome_val = float(outcome_val)

        # Check if param_values is compatible
        for param,values in param_values.iteritems():
            if param not in self.parameters:
                raise ValueError("Parameter '" +param+ "' not valid")
            if values < self.parameters[param]['min'] or values > self.parameters[param]['max']:
                raise ValueError("Parameter '" +param+ "' should have value between "+str(self.parameters[param]['min']) +" and " + str(self.parameters[param]['max']))

        # Check is all parameter values are specified
        for param in self.parameters.keys():
            if param not in param_values:
                raise ValueError("Parameter '" +param+ "' not specified")

        # Check whether this param_values has a results ID
        result_id = self._get_id(param_values)

        if result_id is None:
            # If not, then this is a result that was not suggested,
            # most add it to the server

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
            res = self._client.results().add(self.task_id, True, '', '')
            result_id = res.body['id']

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
                    
            res.body['variables'] = variables

            try:
                res = self._client.result(str(result_id)).update(**res.body)
            except Exception, ex:
                # If the update fails for whatever reason (e.g. validation)
                # we need to make sure we don't leave around empty results
                self._client.result(str(result_id)).delete()
                raise ex

            self._ids_to_param_values[result_id] = param_values

        else:
            # Remove from pending results
            if result_id in self._pending:
                self._pending.remove(result_id)
            
            # Fill in result with the given outcome value
            result = self._client.result(str(result_id)).get().body
            for var in result['variables']:
                if var['name'] == self.outcome_name:
                    var['value'] = outcome_val
                    self._ids_to_outcome_values[result_id] = var
                    break # Assume only one outcome per experiment!
            res = self._client.result(str(result_id)).update(**result)
            self._ids_to_outcome_values[result_id] = outcome_val
        
            
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
            if id in self._pending:
                self._pending.remove(id)

            # Delete from server
            self._client.result(str(id)).delete()
        else:
            print 'Did not find experiment with the provided parameters'


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
    
    def report(self):
        """
        Plot a visual report of the progress made so far in the experiment.
        """

        # Sync with the REST server
        self._sync_with_server()

        # Report historical progress and experiments assumed pending
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
        plt.xlabel('Experiment ID')
        plt.ylabel(self.outcome_name)
        plt.title('Outcome values progression')
        plt.legend(loc=3)
        plt.draw()
        plt.ion()
        plt.show()
        
        # Plot table of experiments
        plt.figure(2)
        param_names = list(np.sort(self.parameters.keys()))
        col_names = param_names + [self.outcome_name]
        cell_text = []
        for id in ids:
            # Get paramater values, put in correct order and add to
            # table with corresponding outcome value
            params, values = zip(*self._ids_to_param_values[id].iteritems())
            s = np.argsort(params)
            values = np.array(values)[s]
            outcome = self._ids_to_outcome_values[id]
            cell_text.append([str(v) for v in values] + [str(outcome)])

        the_table = plt.table(cellText = cell_text, colLabels=col_names, loc='center')

        ## change cell properties
        table_props=the_table.properties()
        table_cells=table_props['child_artists']
        for cell in table_cells:
            cell.set_fontsize(8)

        plt.axis('off')
        plt.title('Table of experiments')
        plt.draw()
        plt.ion()
        plt.show()

        ## Plot the sensitivity to each dimension of the input        
        #sensitivity = self.chooser.get_sensitivity()
        #if sensitivity.shape[0] > 1:
        #   plt.figure(3)
        #   plt.clf()
        #   plt.bar(np.arange(sensitivity.shape[0]), sensitivity)
        #   ax = plt.gca()
        #   labels = []
        #   ax.set_xticks(np.arange(sensitivity.shape[0]))
        #   ax.set_xticklabels(param_names)
        #   plt.title('Relative sensitivity of experiment variables')
        #   plt.draw()
        #   plt.ion()
        #   plt.show()


