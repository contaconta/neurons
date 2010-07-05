if exist('DATASET', 'var')
    switch DATASET
        case 'D1'
            load D1.mat
        case 'D2'
            load D2.mat
        otherwise
            error('Error: no valid DATASET specified.');
    end
else
    disp('...Error, no DATASET specified');
	keyboard;
end

D = single(D);