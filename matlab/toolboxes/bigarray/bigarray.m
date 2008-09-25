%BIGARRAY is a class for storing an array too large for the system memory.
%
%   Bigarray uses file storage to allow you to store and access matrices 
%   too large to fit into memory as if they were stored there.  Data can be 
%   stored as either a series of matlab .mat files or a series of memory 
%   mapped files (memmapfile).  
%
%   To instantiate a bigarray object (and allocated necessary filespace)
%   --------------------------------------------------------------------
%   biga = bigarray(num_rows, num_columns, ...);
%             
%          with the following optional arguments '...':
%          'filename',   filename - filename for storing data on disk
%          'path'    ,   path     - path to a location to store data
%          'bytes'   ,   filesize - sets the size of blocks the data is
%                                   broken into for storing in files and
%                                   for loading into memory.
%          'type'    ,   type     - 'memmapfile' or 'matlab .mat file' 
%                                   determines the method storage method
%
%   To store data to a bigarray object:
%   -----------------------------------
%   biga.store_rows(data, [row_a row_b]); - stores 'data' to bigarray 
%                                           starting at row_a and ending at
%                                           row_b.  Data must be of size
%                                           [row_b-row_a+1 num_columns].
%
%   To retrieve data from a bigarray object:
%   ----------------------------------------
%   data = biga.get_rows([row_a row_b]);  - retrieves elements from bigarray 
%                                           starting at row_a and ending at
%                                           row_b and returns as 'data'.
%
%   To force the bigarray to save data to disk:
%   -------------------------------------------
%   biga.force_save;    - To increase the performance of .mat storage, data
%                         is only written to disk when a new chunk of data
%                         must be loaded, otherwise operations only use the
%                         chunk of data in RAM.  Clearing the bigarray or
%                         exiting matlab may result in data on the disk to
%                         be out of date.  force_save writes data in RAM to
%                         disk it is assured to be current.  This method is
%                         not necessary for memmapfile bigarrays.
%
%   To delete the disk storage files of a bigarray object:
%   ------------------------------------------------------
%   biga.cleanup;       - before clearing a bigarray you may wish to remove
%                         the files it wrote to disk if they are unnecessary.
%
%   TODO: add a method to store columns, or to store rows with some columns
%
%
%   Copyright © 2008 Kevin Smith
%
%   See also MEMMAPFILE



%% class properties defined
classdef bigarray < handle
   properties
        path = [];
        prefix = [];
        bigarray_size = [];
        block_size = [];
        num_blocks = [];
        row_bounds = {};
        filenames = {};
        memmap = [];
        block_data =[];
        loaded_block = [];
        type = [];     % 'matlab .mat file' or 'memmapfile'        
   end% properties

   %% class methods defined 
   methods
        %% the constructor method
        function BA = bigarray(varargin)
            total_rows = varargin{1};
            num_columns = varargin{2};
            BA.prefix = 'BIGARRAY_'; BA.path = [pwd '/'];
            bytesize = 250000000; BA.type = 'matlab .mat file';
            
            for i = 3:nargin
                if strcmp(varargin{i}, 'filename')
                    BA.prefix = varargin{i+1};
                end
                if strcmp(varargin{i}, 'path')
                    BA.path = varargin{i+1};
                end
                if strcmp(varargin{i}, 'bytes')
                    bytesize = varargin{i+1};
                end
                if strcmp(varargin{i}, 'type')
                    BA.type = varargin{i+1};
                end
            end
            
            BA.bigarray_size = [total_rows num_columns];
            rows_per_block = round(bytesize / (num_columns*8));
            BA.block_size = [rows_per_block num_columns];
            BA.num_blocks = ceil(total_rows/rows_per_block);
            
            if BA.num_blocks == 1
                if strcmp(BA.type, 'memmapfile')
                    BA.filenames{1} = [BA.path BA.prefix num2str(1) '.dat'];
                else 
                    BA.filenames{1} = [BA.path BA.prefix num2str(1) '.mat'];
                end
                BA.row_bounds{1} = [1 total_rows];
                BA.allocate(BA.filenames{1});
            else
                for i = 1:BA.num_blocks
                    if i ~= BA.num_blocks
                        if strcmp(BA.type, 'memmapfile')
                            BA.filenames{i} = [BA.path BA.prefix num2str(i) '.dat'];
                        else 
                            BA.filenames{i} = [BA.path BA.prefix num2str(i) '.mat'];
                        end
                        BA.row_bounds{i} = [1 BA.block_size(1)] + BA.block_size(1)*(i-1);
                        BA.allocate(BA.filenames{i});
                    else
                        if strcmp(BA.type, 'memmapfile')
                            BA.filenames{i} = [BA.path BA.prefix num2str(i) '.dat'];
                        else 
                            BA.filenames{i} = [BA.path BA.prefix num2str(i) '.mat'];
                        end
                        previous_row = (i-1)*BA.block_size(1);
                        BA.row_bounds{i} = [1 total_rows - previous_row] + BA.block_size(1)*(i-1);
                        BA.allocate(BA.filenames{i});
                    end
                end
            end
            if strcmp(BA.type, 'memmapfile')
                BA.memmap = memmapfile(BA.filenames{1}, 'Offset', 0, 'Writable', true,      ...    
                'Format', {'double' BA.block_size 'x'});
                BA.block_data = 'unused - needed for matlab .mat type'; BA.loaded_block = 'unused - needed for matlab .mat type';
            else
                load(BA.filenames{1}); BA.block_data = D;  clear D;     
                BA.loaded_block = 1;
                BA.memmap = 'unused - needed for memmapfile type';
            end
        end
       
        %% the get_rows method
        function mat1 = get_rows(varargin)  % (BA, row_lims)
            BA = varargin{1};  row_lims = varargin{2}; nosave = 0;
            if nargin == 3; if strcmp('nosave', varargin{3}); nosave = 1; end; end
            mat1 = []; 
            
            % if only 1 row is requested
            if (row_lims(1) == row_lims(2)) || max(size(row_lims)) == 1
                block = ceil(row_lims(1)/BA.block_size(1));
                r = row_lims(1) - (block-1)*BA.block_size(1);
                if strcmp(BA.type, 'memmapfile')
                    if ~strcmp(BA.memmap.filename, BA.filenames{block})
                        BA.memmap.filename = BA.filenames{block};
                    end
                    mat1 = BA.memmap.data.x(r,:);
                else
                    if BA.loaded_block ~= block
                        if ~nosave
                            D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D;
                        end
                        load(BA.filenames{block}); BA.block_data = D;  clear D;
                        BA.loaded_block = block;
                    end
                    mat1 = BA.block_data(r,:);
                end
                
            % if multiple rows are requested
            else
                firstblock = ceil(row_lims(1)/BA.block_size(1));
                lastblock = ceil(row_lims(2)/BA.block_size(1));
                for i = firstblock:lastblock
                    if (i == firstblock) && (i == lastblock)
                        r1 = row_lims(1) - (i-1)*BA.block_size(1);
                        r2 = row_lims(2) - (i-1)*BA.block_size(1);
                    elseif (i == firstblock)
                        r1 = row_lims(1) - (i-1)*BA.block_size(1);
                        r2 = BA.block_size(1);
                    elseif (i == lastblock)
                        r1 = 1;
                        r2 = row_lims(2) - (i-1)*BA.block_size(1);
                    else
                        r1 = 1;
                        r2 = BA.block_size(1);
                    end

                    if strcmp(BA.type, 'memmapfile')
                        if ~strcmp(BA.memmap.filename, BA.filenames{i})
                            BA.memmap.filename = BA.filenames{i};
                        end
                        mat1 = [mat1; BA.memmap.data.x(r1:r2,:)];
                    else
                        if BA.loaded_block ~= i
                            if ~nosave
                                D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D;
                            end
                            load(BA.filenames{i}); BA.block_data = D;  clear D;
                            BA.loaded_block = i;
                        end
                        mat1 = [mat1; BA.block_data(r1:r2,:)];
                    end
                end
            end
        end

        %% the store_rows method
        function store_rows(BA, mat, row_lims)

            firstblock = ceil(row_lims(1)/BA.block_size(1));
            lastblock = ceil(row_lims(2)/BA.block_size(1));
            mat_top_row = 1;
            
            for i = firstblock:lastblock
                if (i == firstblock) && (i == lastblock)
                    r1 = row_lims(1) - (i-1)*BA.block_size(1);
                    r2 = row_lims(2) - (i-1)*BA.block_size(1);
                elseif (i == firstblock)
                    r1 = row_lims(1) - (i-1)*BA.block_size(1);
                    r2 = BA.block_size(1);
                elseif (i == lastblock)
                    r1 = 1;
                    r2 = row_lims(2) - (i-1)*BA.block_size(1);
                else
                    r1 = 1;
                    r2 = BA.block_size(1);
                end
                if strcmp(BA.type, 'memmapfile')
                    if ~strcmp(BA.memmap.filename, BA.filenames{i})
                        BA.memmap.filename = BA.filenames{i};
                    end
                    BA.memmap.data.x(r1:r2,:) = mat(mat_top_row:mat_top_row+r2-r1,:);
                    mat_top_row = mat_top_row + r2 -r1 + 1;
                else
                    if BA.loaded_block ~= i
                        D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D;
                        load(BA.filenames{i}); BA.block_data = D;  clear D;
                        BA.loaded_block = i;
                    end
                    BA.block_data(r1:r2,:) = mat(mat_top_row:mat_top_row+r2-r1,:);
                    mat_top_row = mat_top_row + r2 -r1 + 1;
                end
            end
        end
        
        %% the force_save method
        function force_save(BA,a) %#ok<INUSD>
            if ~strcmp(BA.type, 'memmapfile')
                D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D; %#ok<NASGU>
            end
        end
        
        %% the cleanup method
        function cleanup(BA)
            if strcmp(BA.type, 'memmapfile')
                for i = 1:length(BA.filenames)
                    cmd = ['rm ' BA.filenames{i}];
                    system(cmd);
                end
            else
                for i = 1:length(BA.filenames)
                    cmd = ['rm ' BA.filenames{i}];
                    system(cmd);
                end
            end
        end
        
        %% the allocate method
        function allocate(BA, filename)
            if strcmp(BA.type, 'memmapfile')
                ELEMENTS_NEEDED = prod(BA.block_size);
                BYTES = ELEMENTS_NEEDED*8;
                if exist(filename, 'file')
                    d = dir(filename);
                    if BYTES == d.bytes
                        disp(['...' filename ' already exists and is correct size!']);
                        return
                    end
                end
                disp(['... allocating memory-mapped file ' filename ]);
                fid = fopen(filename,'w+');
                block = 1000;
                for i = 1:BA.block_size(1)
                    if mod(i,block) ==0
                        fwrite(fid, zeros([block BA.block_size(2)]), 'double'); 
                        ELEMENTS_NEEDED = ELEMENTS_NEEDED - block*BA.block_size(2);
                    end
                end
                fwrite(fid, zeros([ELEMENTS_NEEDED 1]), 'double'); 
                fclose(fid);
            else
                disp(['... allocating matlab .mat file ' filename ]);
                D = zeros(BA.block_size); %#ok<NASGU>
                save(filename, 'D');
            end
        end
        
   end% methods
end% classdef
