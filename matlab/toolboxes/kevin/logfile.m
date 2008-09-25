function DATA = logfile(filenm, action, varargin)
%LOGFILE provides a compact way of writing to and reading from a log file
%
%  [DATA] = logfile(FILENAME, ACTION, ...) requires two inputs: a FILENAME
%  and an ACTION.  The log file itself is composed of rows of data, where 
%  data in each row is delinated by a tab ('\t'). The action can be one of 
%  the following (possibly followed by another argument):
%
%  logfile('l.log', 'header', text)     writes text to the top of the log 
%                                       file. Text can be a string or a
%                                       cell of string (for multiple lines)
%  logfile('l.log', 'column_labels', C) writes column labels C to the log
%                                       file.  C is a cell of strings (one
%                                       to describe each column).
%  logfile('l.log', 'write', ROWDATA)   writes a row vector ROWDATA to the
%                                       log file.
%  logfile('l.log', 'comment', COMMENT) writes a comment to the log file.
%                                       COMMENT can be a string or cell of
%                                       strings for multiple rows.
%  DATA = logfile('l.log', 'read')      reads the data from the log file
%                                       and array DATA.  Ignores the
%                                       header, column labels, and comments.
%  logfile('l.log', 'erase')            erases an existing log file.
%
%
%   Copyright 2008 Kevin Smith
%
%   See also FPRINTF, FWRITE, FREAD, FOPEN, LOAD, SPRINTF

switch action
    case 'write'
        %% write data to the log file
        data = varargin{1};
    
        % get an fid to the file
        fid = fopen(filenm, 'a', 'n');


        % write the data to the next line of the log file
        format = repmat('%0.10g\t', size(data));
        format = [format '\n'];

        fprintf(fid, format, data);

        % close the log file
        fclose(fid);
    
    case 'read'
        %% read data from the log file
        DATA = load(filenm);
        
    case 'column_labels'
        %% write column labels to the log file
        columns = varargin{1};
        column_string = ['%' sprintf('%s\t', columns{:}) sprintf('\n')];
        
        fid = fopen(filenm, 'a', 'n');
        fwrite(fid, column_string);
        fclose(fid);
        
    case 'comment'
        %% write lines of comments to the log file
        columns = varargin{1};
        column_string = [sprintf('%%%10s\n', columns{:}) sprintf('\n')];
        
        fid = fopen(filenm, 'a', 'n');
        fwrite(fid, column_string);
        fclose(fid);    
    
    case 'header'
        %% write a header to the log file
        if exist(filenm, 'file')
            IN = input(['The log file ' filenm ' already exists.  Writing a header will erase its contents.  Procees [Y/n]?'], 's');
            if ~strcmp(IN, 'Y')
                return;
            end
        end
        
            stringcell = varargin{1};

            if ~iscell(stringcell)
                stringcell = {stringcell};
            end
            header_string = sprintf('%% %s\n', stringcell{:});

            fid = fopen(filenm, 'w');
            fwrite(fid, header_string);
            fclose(fid);
        
    case 'erase'
        %% erase the log file
        system(['rm ' filenm]);
        
    otherwise
        %% no action specified
        error('Please specify either "read", "write", "erase", etc as second argument (see "help logfile" for details).');
    
end