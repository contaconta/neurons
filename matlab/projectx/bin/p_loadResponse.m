function data = p_loadResponse(index, learner_type)

% 'HA'

% call the mex function to store the data to the server
data = mexLoadResponse(index, learner_type);