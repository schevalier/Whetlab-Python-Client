from nose.tools import *
import whetlab, whetlab.server
from time import time, sleep
from nose.tools import with_setup, assert_equals
import numpy as np
import numpy.random as npr

whetlab.RETRY_TIMES = [] # So that it doesn't wait forever for tests that raise errors

default_access_token = None

default_description = ''
default_parameters = { 'p1':{'type':'float', 'min':0, 'max':10.0, 'size':1},
                            'p2':{'type':'integer', 'min':0, 'max':10, 'size':1}}
default_outcome = {'name':'Dummy outcome'}

last_created_experiment = ""

def test_required_prop_are_supported():
    """ All required properties should be supported, for parameters and outcome. """
    
    # Parameters
    for props in whetlab.required_properties.values():
        for x in props:
            assert( x in whetlab.supported_properties )

    # Outcome
    for x in whetlab.outcome_required_properties:
        assert( x in whetlab.outcome_supported_properties )

def test_default_values_are_legal():
    """ All default values for properties should be legal, for parameters and outcome. """
    
    #Parameters
    for k,v in whetlab.default_values.items():
        if k in whetlab.legal_values:
            assert( v in whetlab.legal_values[k] )

    # Outcome
    for k,v in whetlab.outcome_default_values.items():
        if k in whetlab.outcome_legal_values:
            assert( v in whetlab.outcome_legal_values[k] )

def test_delete_experiment():
    """ Delete experiment should remove the experiment from the server. """
    
    name = 'test ' + str(time())
    scientist = whetlab.Experiment(access_token=default_access_token,
                                   name=name,
                                   description=default_description,
                                   parameters=default_parameters,
                                   outcome=default_outcome)
    
    scientist.update({'p1':5.,'p2':1},5)

    # Delete experiment
    whetlab.delete_experiment(name,default_access_token)

    # Should now be possible to create an experiment with the same name
    scientist = whetlab.Experiment(access_token=default_access_token,
                                   name=name,
                                   description=default_description,
                                   parameters=default_parameters,
                                   outcome=default_outcome)
    
    # Re-deleting it
    whetlab.delete_experiment(name,default_access_token)


def setup_function(): 
    try:
      whetlab.delete_experiment('test_experiment',default_access_token)
    except:      
      pass
 
def teardown_function():
    whetlab.delete_experiment('test_experiment',default_access_token)

class TestExperiment:

    def __init__(self):
      self.name = 'test ' + str(time())

    # Before running each test make sure that there is no experiment
    # with this name
    def setup(self):
      try:
        whetlab.delete_experiment(self.name,default_access_token)
      except:      
        pass

    # Make sure to clean up any created experiments with this name
    def teardown(self):
      try:
        whetlab.delete_experiment(self.name,default_access_token)
      except:      
        pass

    @raises(whetlab.server.error.client_error.ClientError)
    def test_same_name(self):
        """ Can't create two experiments with same name (when resume is False). """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        # Repeat Experiment creation to raise error, with resume set to False
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description+'2',
                                       parameters=default_parameters,
                                       outcome=default_outcome,
                                       resume = False)

    def test_resume_false(self):
        """ If resume is False and experiment's name is unique, can create an experiment. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome,
                                       resume = False)
              
    def test_resume(self):
        """ Resume correctly loads previous results. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':2.1,'p2':1},3)

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description)
        # Make sure result is still there
        assert( cmp(scientist._ids_to_param_values.values()[0],{'p1':2.1,'p2':1}) == 0 )
        assert( cmp(scientist._ids_to_outcome_values.values()[0],3) == 0 )

    @raises(ValueError)
    def test_empty_name(self):
        """ Experiment's name can't be empty. """

        name = ''
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

    @raises(whetlab.server.error.client_error.ClientError)
    def test_name_too_long(self):
        """ Experiment's name must have at most 500 caracters. """

        name = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)
    
#    @raises(whetlab.server.error.client_error.ClientError)
#    def test_description_too_long(self):
#        """ Experiment's description must have at most 500 caracters. """
#
#        name = 'test ' + str(time())
#        description = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#        scientist = whetlab.Experiment(access_token=default_access_token,
#                                       name=name,
#                                       description=description,
#                                       parameters=default_parameters,
#                                       outcome=default_outcome)



    @raises(ValueError)
    def test_empty_parameters(self):
        """ Experiment's parameters can't be empty. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters={},
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_empty_outcome(self):
        """ Experiment's outcome can't be empty. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome={})

    @raises(ValueError)
    def test_unknown_parameter_properties(self):
        """ Parameter properties must be valid. """

        bad_parameters = { 'p1':{'type':'float', 'min':0, 'max':10.0, 'size':1, 'fake_property':10}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_min_max_properties(self):
        """ Parameter property 'min' must be smaller than 'max'. """

        bad_parameters = { 'p1':{'type':'float', 'min':10., 'max':1., 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_float_for_int_bounds(self):
        """ Parameter properties 'min' and 'max' must be integers if the parameter is an integer. """

        bad_parameters = { 'p1':{'type':'integer', 'min':0.0, 'max':0.5, 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_nan_bounds(self):
        """ Parameter properties 'min' and 'max' must be finite. """

        bad_parameters = { 'p1':{'type':'float', 'min':np.nan, 'max':0.5, 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_infinite_bounds(self):
        """ Parameter properties 'min' and 'max' must be finite. """

        bad_parameters = { 'p1':{'type':'float', 'min':-np.inf, 'max':np.inf, 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_large_neg_bounds(self):
        """ Parameter properties 'min' and 'max' must be greater than -1e32. """

        bad_parameters = { 'p1':{'type':'integer', 'min':-1e35, 'max':1.0, 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_large_bounds(self):
        """ Parameter properties 'min' and 'max' must be less than 1e32. """

        bad_parameters = { 'p1':{'type':'integer', 'min':1.0, 'max':1e33, 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_vector_bounds(self):
        """ Parameter properties 'min' and 'max' must be finite numbers. """

        bad_parameters = { 'p1':{'type':'float', 'min':[0.1, 0.2], 'max':0.5, 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_legal_property_value(self):
        """ Parameter property must take a legal value. """

        bad_parameters = { 'p1':{'type':'BAD_VALUE', 'min':1., 'max':10., 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_enum_not_supported(self):
        """ Parameter type 'enum' not yet supported. """

        bad_parameters = { 'p1':{'type':'enum', 'min':1., 'max':10., 'size':1}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    def test_good_enum_options(self):
        """ Enum options with a legal name. """

        bad_parameters = { 'p1':{'type':'enum', 'options':['one', 'two', 'three']}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    def test_enum_with_two_options(self):
        """ Enums should work with just two options. """

        bad_parameters = { 'p1':{'type':'enum', 'options':['one', 'two']}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)        

    def test_enum_update(self):
        """ Update supports enum. """

        parameters = { 'p1':{'type':'enum', 'options':['one', 'two']}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)        

        scientist.update({'p1':'one'},10)

    @raises(ValueError)
    def test_bad_enum_options(self):
        """ Enum options must take a legal name. """

        bad_parameters = { 'p1':{'type':'enum', 'options':['1','2','3']}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)

    @raises(ValueError)
    def test_bad_enum_update(self):
        """ Enum can't update with value not in options. """

        bad_parameters = { 'p1':{'type':'enum', 'options':['one', 'two', 'three']}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)
        scientist.update({'p1':'four'},10.)

    def test_list_enum_update(self):
        """ Update supports list of enums (size > 1). """

        parameters = { 'p1':{'type':'enum', 'options':['one', 'two', 'three'], 'size':3}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)        
        job =  {u'p1':['three','one','one']}
        scientist.update(job,10.)

    def test_list_enum_suggest(self):
        """ Suggest supports list of enums (size > 1). """

        parameters = { 'p1':{'type':'enum', 'options':['one', 'two'], 'size':3}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)        

        job = scientist.suggest()
        assert(len(job['p1']) == 3)
        assert(job['p1'][0] in {'one','two'})
        assert(job['p1'][1] in {'one','two'})
        assert(job['p1'][2] in {'one','two'})

    @raises(ValueError)
    def test_list_bad_enum_update(self):
        """ Enum can't update when one value in list (size>1) not in options. """

        bad_parameters = { 'p1':{'type':'enum', 'options':['one', 'two', 'three'],'size':4}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=bad_parameters,
                                       outcome=default_outcome)
        scientist.update({'p1':['one','two','one','four']},10.)

    @raises(whetlab.server.error.client_error.ClientError)
    def test_access_token(self):
        """ Valid access token must be provided. """

        scientist = whetlab.Experiment(access_token='',
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

    def test_cancel(self):
        """ Cancel removes a result. """

        name = 'test test_cancel ' + str(time())
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':5.1,'p2':5},10)
        scientist.cancel({'p1':5.1,'p2':5})
        
        # Make sure result was removed
        scientist._sync_with_server()
        assert( len(scientist._ids_to_param_values) == 0 )
        assert( len(scientist._ids_to_outcome_values) == 0 )

    def test_get_by_result_id(self):
        """ Get a result by the id. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        jobs = []
        for i in xrange(5):
          jobs.append(scientist.suggest())

        for i in xrange(5):
          scientist.update(jobs[i], np.random.randn())

        # Make sure result was removed
        scientist._sync_with_server()

        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          job = scientist.get_by_result_id(result_id)
          assert_equals(job, jobs[i])

    def test_get_result_id(self):
        """ Get the id associated with a set of parameters. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        jobs = []
        for i in xrange(5):
          jobs.append(scientist.suggest())

        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          j = dict(jobs[i]) # Throw away the id
          assert_equals(result_id, scientist.get_id(j))
          assert_equals(scientist.get_id(jobs[i]), scientist.get_id(j))
          scientist.update_by_result_id(result_id, npr.randn())
          assert_equals(result_id, scientist.get_id(j))
          assert_equals(scientist.get_id(jobs[i]), scientist.get_id(j))

    def test_cancel_by_result_id(self):
        """ Cancel removes a result. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        jobs = []
        for i in xrange(5):
          jobs.append(scientist.suggest())

        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          scientist.update_by_result_id(result_id, npr.randn())

        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          scientist.cancel_by_result_id(result_id)

        # Make sure result was removed
        scientist._sync_with_server()
        assert_equals(len(scientist._ids_to_param_values), 0 )
        assert_equals(len(scientist._ids_to_outcome_values), 0 )

    def test_get_all_results(self):
        """ Cancel removes a result. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        def count_in_list(j, jobs):
          hits = 0
          for job in jobs:
            if scientist.get_id(job) == scientist.get_id(j):
              hits += 1
          return hits

        jobs = []
        for i in xrange(5):
          jobs.append(scientist.suggest())
          j, o = scientist.get_all_results()
          assert_equals(len(j), len(o))
          assert_equals(len(j), i+1)
          assert_equals(o[i], None)
          assert_equals(count_in_list(jobs[i], j), 1)

        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          outcome = npr.randn()
          scientist.update_by_result_id(result_id, outcome)
          j, o = scientist.get_all_results()
          assert_equals(len(j), len(o))
          assert_equals(count_in_list(jobs[i], j), 1)
          assert(outcome in o)

        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          scientist.cancel_by_result_id(result_id)
          j, o = scientist.get_all_results()
          assert_equals(len(j), len(o))
          assert_equals(count_in_list(jobs[i], j), 0)
          assert_equals(len(j), 5-i-1)

    def test_update_by_result_id(self):
        """ Update adds and can overwrite a result. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        jobs = []
        for i in xrange(5):
          jobs.append(scientist.suggest())

        outcomes = []
        for i in xrange(5):          
          result_id = scientist.get_id(jobs[i])
          outcomes.append(npr.randn())
          scientist.update_by_result_id(result_id, outcomes[-1])

        # Make sure result was added
        scientist._sync_with_server()
        for i in xrange(5):
          result_id = scientist.get_id(jobs[i])
          assert_equals(scientist._ids_to_outcome_values[result_id], outcomes[i])

    def test_update(self):
        """ Update adds and can overwrite a result. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':5.1,'p2':5},10)

        # Make sure result was added
        scientist._sync_with_server()
        assert( cmp(scientist._ids_to_param_values.values()[0],{'p1':5.1,'p2':5}) == 0 )
        assert( cmp(scientist._ids_to_outcome_values.values()[0],10) == 0 )

        # Make sure result was overwritten
        scientist.update({'p1':5.1,'p2':5},20)
        scientist._sync_with_server()
        assert( len(scientist._ids_to_param_values.values()) == 1 )
        assert( len(scientist._ids_to_outcome_values.values()) == 1 )
        assert( cmp(scientist._ids_to_param_values.values()[0],{'p1':5.1,'p2':5}) == 0 )
        assert( cmp(scientist._ids_to_outcome_values.values()[0],20) == 0 )

    def test_suggest_twice(self):
        """ Calling suggest twice returns two different jobs. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        a = scientist.suggest()
        sleep(2)
        b = scientist.suggest()
        
        # Two suggested jobs are different
        assert( cmp(a,b) != 0 )

    def test_suggest(self):
        """ Suggest return a valid job. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        a = scientist.suggest()

        # Check all parameter names are valid in suggestion
        for k in a.keys():
            assert(k in default_parameters.keys())

        # Check if all parameters were assigned a value
        for k in default_parameters.keys():
            assert(k in a)

        # Check parameter values are within the min/max bounds
        for k,v in a.items():
            assert(v >= default_parameters[k]['min'])
            assert(v <= default_parameters[k]['max'])

        # Check parameter values are of right type
        for k,v in a.items():
            if default_parameters[k]['type'] == 'integer':
                assert(type(v) == int)
            if default_parameters[k]['type'] == 'float':
                assert(type(v) == float)
        
    def test_best(self):
        """ Best returns the best job. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':1.,'p2':4},1)
        scientist.update({'p1':4.,'p2':2},2)
        scientist.update({'p1':5.,'p2':1},1000)
        scientist.update({'p1':9.,'p2':9},3)
        scientist.update({'p1':1.,'p2':1},4)
        scientist.update({'p1':5.,'p2':5},5)

        assert(cmp(scientist.best(),{'p1':5.,'p2':1})==0)

    def test_best_with_nan(self):
        """ Best returns the best job. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':1.,'p2':4},1.0)
        scientist.update({'p1':4.,'p2':2},2.0)
        scientist.update({'p1':5.,'p2':1},1000)
        scientist.update({'p1':9.,'p2':9},3)
        scientist.update({'p1':1.,'p2':1},4)
        scientist.update({'p1':5.,'p2':5},5)
        scientist.update({'p1':5.,'p2':2}, np.nan)
        scientist.update({'p1':5.,'p2':7}, np.nan)

        assert(cmp(scientist.best(),{'p1':5.,'p2':1})==0)


    def test_pending(self):
        """ Pending returns jobs that have not been updated. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        a = scientist.suggest()
        b = scientist.suggest()
        c = scientist.suggest()
        scientist.update(b,10)
        l = scientist.pending()

        assert(a in l)
        assert(b not in l)
        assert(c in l)
        assert(len(l) == 2)

    def test_clear_pending(self):
        """ Should remove pending jobs only. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        a = scientist.suggest()
        b = scientist.suggest()
        c = scientist.suggest()
        scientist.update(b,10)
        scientist.clear_pending()

        assert( len(scientist.pending()) == 0 )

        # Make sure only result left is "b"
        scientist._sync_with_server()
        print scientist._ids_to_param_values
        assert( cmp(scientist._ids_to_param_values.values()[0],b) == 0 )
        assert( cmp(scientist._ids_to_outcome_values.values()[0],10) == 0 )

    @raises(ValueError)
    def test_update_parameter_too_small(self):
        """ Update should raise error if parameter smaller than minimum. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':-5.,'p2':5},5)

    @raises(ValueError)
    def test_update_parameter_too_big(self):
        """ Update should raise error if parameter larger than maximum. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':5.,'p2':50},5)
    
    @raises(TypeError)
    def test_update_parameter_not_integer(self):
        """ Update should raise error if an integer parameter has a non-integer value. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':5.,'p2':1.},5)

    @raises(TypeError)
    def test_update_parameter_not_float(self):
        """ Update should raise error if a float parameter has a non-float value. """

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=default_parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':5,'p2':1},5)

    def test_create_experiment_with_defaults(self):
        """ Can create experiment with floats/integers simply by specifying min/max."""

        minimal_parameter_description = {'p1':{'min':0,'max':10}, 'p2':{'type':'integer', 'min':1, 'max':4}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=minimal_parameters_description,
                                       outcome=default_outcome)


    def test_create_experiment_with_defaults(self):
        """ Can create experiment with floats/integers simply by specifying min/max."""

        minimal_parameters_description = {'p1':{'min':0,'max':10}, 'p2':{'type':'integer', 'min':1, 'max':4}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=minimal_parameters_description,
                                       outcome=default_outcome)


    def test_int_instead_of_integer(self):
        """ Can use 'int' as type, instead of 'integer'."""

        parameters = {'p1':{'type':'int', 'min':1, 'max':4}}
        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)


    def test_multidimensional_parameters(self):
        """ Can use multidimensional parameters."""

        parameters = { 'p1':{'type':'float', 'min':0, 'max':10.0, 'size':3},
                       'p2':{'type':'integer', 'min':0, 'max':10, 'size':5}}

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)

        job = scientist.suggest()
        assert(len(job['p1']) == 3)
        assert(len(job['p2']) == 5)
        scientist.update({'p1':[1.2,2.3,3.4],'p2':[4,2,5,2,1]},10.2)
        
    @raises(ValueError)
    def test_multidimensional_correct_size(self):
        """ Parameters must have correct size."""

        parameters = { 'p1':{'type':'float', 'min':0, 'max':10.0, 'size':3},
                       'p2':{'type':'integer', 'min':0, 'max':10, 'size':5}}

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':[1.2,2.3,3.4],'p2':[4,2,5]},10.2)
        
    @raises(ValueError)
    def test_multidimensional_min_max(self):
        """ All dimensions must be within min/max bounds."""

        parameters = { 'p1':{'type':'float', 'min':0, 'max':10.0, 'size':3},
                       'p2':{'type':'integer', 'min':0, 'max':10, 'size':5}}

        scientist = whetlab.Experiment(access_token=default_access_token,
                                       name=self.name,
                                       description=default_description,
                                       parameters=parameters,
                                       outcome=default_outcome)

        scientist.update({'p1':[1.2,2.3,3.4],'p2':[4,2,5,12,3]},10.2)
        
