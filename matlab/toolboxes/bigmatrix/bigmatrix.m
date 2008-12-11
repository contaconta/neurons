%BIGMATRIX is a class for storing an matrix too large for the system memory.
%
%   note: cols are faster, but must be sequential.  rows are slower, but
%   can have breaks row_inds = [1 2 5 6]
%
%   B = bigmatrix(2000, 60000, 'filename', 'temp.dat', 'memory', 100000000, 'precision', 'single');
%   B.store(data, row, col);            % stores a single element
%   B.storeBlock(data, rows, cols);     % must be consecutive rows, cols!
%   B.storeCols(data, cols);            % must be consecutive rows, cols!
%   B.storeRows(data, rows);            % must be consecutive rows, cols!
%   data = B.get(row, col);             % gets a single element
%   data = B.getCols(cols);             % gets a column (or list of columns)
%   data = B.getRows(rows);             % gets a row (or list of rows)
%
%   Copyright © 2008 Kevin Smith
%
%   See also BIGARRAY



%% class properties %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef bigmatrix < handle
   properties
        filename = 'temp.dat';                  % name of the file to store the BIGMATRIX (default = temp.dat)
        rows = 0;                               % the number of rows in the BIGMATRIX (default = 0)
        cols = 0;                               % the number of cols in the BIGMATRIX (default = 0)
        memory_footprint = 333000000;           % the amount of RAM the BIGMATRIX can use (default 333MB)
        precision = 'double'                    % the precision of the data stored in the BIGMATRIX (default = 'double')
        bytes = 8;                              % the number of bytes used for this precision (double = 8, single = 4)
   end
   properties (SetObservable = true)
        rowCache = [];                          % a cache of recently accessed rows of the BIGMATRIX
        colCache = [];                          % a cache of recently accessed columns of the BIGMATRIX
        colCacheLims = [1 1];                   % the first and last col indexes in the colCache
        rowCacheLims = [1 1];                   % the first and last row indexes in the rowCache
        refreshRowCacheFlag = 0;                % flag indicating that the rowCache needs to be refreshed
        refreshColCacheFlag = 0;                % flag indicating that the colCache needs to be refreshed
        fid =[];                                % the file identifier for the storage file
   end% properties

   
   
   methods
       
        %% the constructor method %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Constructs a new BIGMATRIX object.  User must specify the number
        % of ROWS and the number of COLS that the BIGMATRIX storage file
        % will contain.  This number cannot currently be changed.
        % Opionally, the filename of the storage file may be specified, the
        % precision of the data, and the amount of RAM the BIGMATRIX
        % should consume may be specified.
        function obj = bigmatrix(varargin)
            obj.rows = varargin{1}; obj.cols = varargin{2};
            NEW = 1;
            
            for i = 3:nargin
                if strcmp(varargin{i}, 'filename')
                    obj.filename = varargin{i+1};
                end
                if strcmp(varargin{i}, 'memory')
                    obj.memory_footprint = varargin{i+1};
                end
                if strcmp(varargin{i}, 'precision')
                    obj.precision = varargin{i+1};
                    switch obj.precision
                        case 'single'
                            obj.bytes = 4;
                        case 'double'
                            obj.bytes = 8;
                        otherwise 
                            error('unsupported precision type');
                    end                    
                end
                if strcmp(varargin{i}, 'open');
                    NEW = 0;
                end
            end
            
            % we are creating a new storage file.
            if NEW
                obj.fid = fopen(obj.filename, 'w');
                frewind(obj.fid);

                % if it is a new file, fill the file with zeros
                bytes_needed = obj.rows * obj.cols * obj.bytes;
                bytes_used = 0;

                % if the memory footprint is not enough to store the whole
                % array, break it into chunks and write zeros to the file.
                if obj.memory_footprint < bytes_needed

                    while bytes_used < bytes_needed

                        bytes_to_write = min(obj.memory_footprint, bytes_needed - bytes_used);

                        A = zeros([bytes_to_write/obj.bytes 1]);
                        fwrite(obj.fid, A, obj.precision);

                        disp(['    ...writing ' num2str(bytes_to_write) ' bytes to ' obj.filename ]);

                        bytes_used = bytes_used + bytes_to_write;
                        clear A;
                    end
                else
                    A = zeros(obj.rows, obj.cols, obj.precision);   
                    fwrite(obj.fid, A, 'float');
                    obj.data = A;
                end
                
                obj.rowCache = zeros(1,obj.cols);
                obj.colCache = zeros(obj.rows,1);
                fclose(obj.fid);   
            
            % we are opening a previously created storage file.    
            else
                obj.rowCache = obj.getRowsFromFile(1);
                obj.colCache = obj.getColsFromFile(1);
                obj.fid = fopen(obj.filename, 'r');
                fclose(obj.fid);
            end
        end
            
        %% store (single element) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Stores a single element specified by DATA to the storage file at
        % the location specified by row, col.
        function store(obj, data, row, col)
            if ~isequal(size(data), [length(row) length(col)])
                error('error: the size of the data is not [1 1]');
            end
            if col > obj.cols
                error('error: the col index is out of bounds');
            end
            if row > obj.rows
                error('error: the row index is out of bounds');
            end
           
            obj.fid = fopen(obj.filename, 'r+');
            position = ( (row-1) + obj.rows*(col-1) )* obj.bytes;
            fseek(obj.fid, position, 'bof');
            fwrite(obj.fid, data, obj.precision);
            
            fclose(obj.fid);
            
            obj.refreshRowCacheFlag = 1;         % we have changed stored data, cache should be refreshed
            obj.refreshColCacheFlag = 1;
        end
        
        %% storeRows %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Stores row data contained in DATA to the CONSECUTIVE rows of the storage file
        % specified by row_inds.
        function storeRows(obj, data, row_inds)
            
            if size(data,1) ~= length(row_inds)
                error('error: the number of rows in data and row_inds do not match');
            end
            
            if size(data,2) ~= obj.cols
                error('error: the number of cols in data does not match the number of cols in the bigmatrix');
            end
            
            
            % store it to the file
            obj.fid = fopen(obj.filename, 'r+');
            
            for i = 1:length(row_inds);
                seek_bytes = (row_inds(i)-1)*obj.bytes;
                fseek(obj.fid, seek_bytes, 'bof');
                fwrite(obj.fid, data(i,1), obj.precision);
                fwrite(obj.fid, data(i,2:size(data,2)), obj.precision, (obj.rows-1)*obj.bytes);
            end

            fclose(obj.fid);
            obj.refreshRowCacheFlag = 1;         % we have changed stored data, cache should be refreshed
            obj.refreshColCacheFlag = 1;
        end
        
        
        %% storeCols %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Stores column data contained in DATA to CONSECUTIVE columns of the
        % storage file specified by col_inds.
        function storeCols(obj, data, col_inds )
            
            if size(data,2) ~= length(col_inds)
                error('error: the number of columns in data and col_inds do not match');
            end
            
            if size(data,1) ~= obj.rows
                error('error: the number of rows in data does not match the number of rows in the bigmatrix');
            end
            
            % store it to the file
            obj.fid = fopen(obj.filename, 'r+');
            fseek(obj.fid, (col_inds(1)-1)*obj.rows*obj.bytes, 'bof');
            fwrite(obj.fid, data, obj.precision);
            fclose(obj.fid);
            obj.refreshRowCacheFlag = 1;         % we have changed stored data, cache should be refreshed
            obj.refreshColCacheFlag = 1;
        end
        
        
        %% storeBlock %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Stores a block of data contained in DATA to the CONSECUTIVE rows and columns
        % specified by row_inds and col_inds
        function storeBlock(obj, data, row_inds, col_inds)
            
            if ~isequal(size(data), [length(row_inds) length(col_inds)])
                error('error: the size of the data does not match the row and column indexes');
            end
            
            if (row_inds(1) < 1) || (row_inds(length(row_inds)) > obj.rows)
                error('error: the row_inds do not fit in the bounds of the bigmatrix.');
            end
            
            if (col_inds(1) < 1) || (col_inds(length(col_inds)) > obj.cols)
                error('error: the col_inds do not fit in the bounds of the bigmatrix.');
            end
            
            obj.fid = fopen(obj.filename, 'r+');
            
            for j = 1:length(col_inds);
                seek_bytes = (col_inds(j)-1)*obj.rows*obj.bytes + (row_inds(1)-1)*obj.bytes;
                fseek(obj.fid, seek_bytes, 'bof');
                coldata = data(:,j);
                fwrite(obj.fid, coldata, obj.precision);
            end
            
            fclose(obj.fid);
            obj.refreshRowCacheFlag = 1;         % we have changed stored data, cache should be refreshed
            obj.refreshColCacheFlag = 1;
        end
        
        
        
        %% get (single element) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Retrieves a single element specified by row, col from the storage 
        % file and returns it as DATA.
        function data = getdataFromFile(obj, row, col)
        
            if col > obj.cols
                error('error: the col index is out of bounds');
            end
            if row > obj.rows
                error('error: the row index is out of bounds');
            end
            
            % update the caches
            obj.updateColCache(col);
            obj.updateRowCache(row);
           
            if inColCache(col)
                % it's in the colCache, don't need to do a file read
                data = obj.colCache(row, col - obj.colCacheLims(1) + 1);
                
            elseif inRowCache(row)
                % it's in the rowCache, don't need to do a file read
                data = obj.rowCache(row - obj.rowCacheLims(1) + 1, col);
                
            else     
                % otherwise read it from the file
                obj.fid = fopen(obj.filename, 'r+');
                position = ( (row-1) + obj.rows*(col-1) )* obj.bytes;
                fseek(obj.fid, position, 'bof');
                data = fread(obj.fid, 1, obj.precision);
            end
            
            fclose(obj.fid);
        end
        
        %% getColsFromFile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Retrieves columns specified by col_inds from the storage file and
        % returns them as DATA.
        function data = getColsFromFile(obj, col_inds)
            
            obj.fid = fopen(obj.filename, 'r+');
            fseek(obj.fid, (col_inds(1)-1)*obj.rows*obj.bytes, 'bof'); 
            data = fread(obj.fid, [obj.rows length(col_inds)], obj.precision);
            fclose(obj.fid);
        end
        
        %% updateColCache %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Updates the columns cache so that the block containing "col" is
        % now in the columns cache.
        function updateColCache(obj,col)
            
            if ~obj.inColCache(col)  || obj.refreshColCacheFlag
                if (col > obj.cols) || (col < 1)
                    error(['error: row is out of bounds. bigmatrix dims [' num2str(obj.rows) ' ' num2str(obj.cols) ']']);     
                end

                % determine the size of the cache, the first and last index
                block = round ( (obj.memory_footprint/2) / (obj.rows * obj.bytes) );
                blocks = 1:block: obj.cols;
                [h,ind] = find(1:block:obj.cols <= col, 1, 'last');
                first = blocks(ind);
                last = min(first + block - 1, obj.cols);

                % update the cache with data from the storage file
                %disp(['updating the colCache with cols ' num2str(first) ':' num2str(last)]);
                obj.colCache = obj.getColsFromFile(first:last);
                obj.colCacheLims = [first last];
                obj.refreshColCacheFlag = 0;
            else
                %disp(['col ' num2str(col) ' is already in the cache.']);
            end
        end
        
        %% inColCache %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Returns 1 if the requested col is currently in the columns cache, 
        % a 0 if it is not.
        function B = inColCache(obj, col)
            
            lowlim = ones(size(col))*obj.colCacheLims(1);
            upperlim = ones(size(col))*obj.colCacheLims(2);
            
            B = (lowlim <= col) & (col <= upperlim);
        end
        
        
        %% getCols %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data = getCols(obj, col_inds)
 
            % allocate space for DATA
            data = zeros(obj.rows, length(col_inds));
            
            % update the colCache so it contains the first column we want
            obj.updateColCache(col_inds(1));
            
            % get the columns in the colCache and put them in DATA
            data_cols = obj.inColCache(col_inds);
            if ~isempty(find(data_cols,1))
                cached_cols = col_inds(data_cols);
                data(:,data_cols) = obj.colCache(:,cached_cols - obj.colCacheLims(1) +1);
                %disp(['columns [' num2str(cached_cols) '] found in the colCache']);
            end
            
            % retrieve the missing columns from the file and put them in DATA
            if ~isempty(find(~data_cols,1))
                file_col_inds = col_inds(~data_cols);
                data(:, ~data_cols) = obj.getColsFromFile(file_col_inds);
                %disp(['columns [' num2str(file_col_inds) '] retrieved from ' num2str(obj.filename)]);
            end
            
            % PREVIOUSLY UPDATED THE CACHE AFTER GETTING THE DATA
%             % update the colCache to it contains the last column we
%             % grabbed from the file
%             obj.updateColCache(col_inds(length(col_inds)));
            
        end
        
        %% getRowsFromFile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Retrieves the rows specified by row_inds from the storage file
        % and returns them as DATA.  YOU SHOULD NEVER USE THIS METHOD FOR
        % MANY REPEATED READS - IT IS MUCH FASTER TO USE GETROWS, WHICH
        % USES THE CACHE AND READS DATA MUCH FASTER VIA GETCOLSFROMFILE
        function data = getRowsFromFile(obj, row_inds)
            
            obj.fid = fopen(obj.filename, 'r+');
            
            data = zeros([length(row_inds), obj.cols], obj.precision);
            
            for i = 1:length(row_inds);
                seek_bytes = (row_inds(i)-1)*obj.bytes;
                fseek(obj.fid, seek_bytes, 'bof');
                data(i,:) = fread(obj.fid, [1 obj.cols], obj.precision, (obj.rows-1)*obj.bytes);
            end
            
            fclose(obj.fid);
        end
        
        %% updateRowCache %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function updateRowCache(obj, row)
        
            if ~obj.inRowCache(row) || obj.refreshRowCacheFlag
                if (row > obj.rows) || (row < 1)
                    error(['error: row is out of bounds. bigmatrix dims [' num2str(obj.rows) ' ' num2str(obj.cols) ']']);
                end

                % determine the size of the cache, the first and last row index
                block = round ( (obj.memory_footprint/2) / (obj.cols * obj.bytes) );
                blocks = 1:block: obj.rows;
                [h,ind] = find(1:block:obj.rows <= row, 1, 'last');
                first_row = blocks(ind);
                last_row = min(first_row + block - 1, obj.rows);

                % because it is faster to read in columns, we will scan the
                % file in columns and fill in the appropriate rows as we go.
                
                % first we need to determine the # of columns we can fit in memory
                block = round ( (obj.memory_footprint/2) / (obj.rows * obj.bytes) );
                blocks = 1:block: obj.cols;
                
                for i = blocks
                    first_col = i;
                    last_col = min(first_col + block - 1, obj.cols);
                    %disp(['reading cols [' num2str(first_col) ':' num2str(last_col) ']' ]);
                    obj.colCache = obj.getColsFromFile(first_col:last_col);
                    obj.colCacheLims = [first_col last_col];
                    
                    obj.rowCache(1:last_row-first_row+1, first_col:last_col) = obj.colCache(first_row:last_row,:);
                end
                
                % update the cache with data from the storage file
                %disp(['updating the rowCache with rows ' num2str(first_row) ':' num2str(last_row)]);
                obj.rowCacheLims = [first_row last_row];
                obj.refreshRowCacheFlag = 0;
                obj.refreshColCacheFlag = 0;
            else
                %disp(['row ' num2str(row) ' is already in the cache.']);
            end
        
        end
        
        
        %% inRowCache %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        % Returns 1's if the requested rows are currently in the rows cache, 
        % or 0's if they are not.
        function B = inRowCache(obj,rows)
            lowlim = ones(size(rows))*obj.rowCacheLims(1);
            upperlim = ones(size(rows))*obj.rowCacheLims(2);
            
            B = (lowlim <= rows) & (rows <= upperlim);
        end


        %% getRows %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function data = getRows(obj, row_inds)
 
            % allocate space for DATA
            data = zeros(length(row_inds), obj.cols);
            
            % update the rowCache so it contains the first row we want
            obj.updateRowCache(row_inds(1));
            
            % get the columns in the colCache and put them in DATA
            data_rows = obj.inRowCache(row_inds);
            if ~isempty(find(data_rows,1))
                cached_rows = row_inds(data_rows);
                data(data_rows,:) = obj.rowCache(cached_rows - obj.rowCacheLims(1) +1,:);
                %disp(['rows [' num2str(cached_rows) '] found in the rowCache']);
            end
            
            % retrieve the missing columns from the file and put them in DATA
            if ~isempty(find(~data_rows,1))
                file_row_inds = row_inds(~data_rows);
                data(~data_rows,:) = obj.getRowsFromFile(file_row_inds);
                %disp(['rows [' num2str(file_row_inds) '] retrieved from ' num2str(obj.filename)]);
            end
        end
        
        
        
        %% get %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % a simple function to get a single entry from the
        % bigmatrix
        function data = get(obj, row, col)
            
            % update the colCache so it contains the first column we want
            obj.updateColCache(col);
            
            % get the columns in the colCache and put them in DATA
            data_col = obj.inColCache(col);
            if ~isempty(find(data_col,1))
                cached_cols = col(data_col);
                data = obj.colCache(row,cached_cols - obj.colCacheLims(1) +1);
            end
            
            % retrieve the missing columns from the file and put them in DATA
            if ~isempty(find(~data_col,1))
                data = getdataFromFile(row, col);
            end
            
        end
        

   end %methods
end %classdef
       
