%BIGMATRIX is a class for storing an matrix too large for the system memory.
%
%   note: cols are faster, but must be sequential.  rows are slower, but
%   can have breaks row_inds = [1 2 5 6]
%
%
%   Copyright © 2008 Kevin Smith
%
%   See also BIGARRAY



%% class properties defined
classdef bigmatrix 
   properties
        filename = []
%         bigarray_size = [];
%         block_size = [];
%         num_blocks = [];
%         row_bounds = {};
        rows = 0;
        cols = 0;
        memory_footprint = 333000000;
        data = [];
        fid =[];
     
   end% properties

   
   %% class methods defined 
   methods
       
        %% the constructor method
        function obj = bigmatrix(varargin)
            obj.rows = varargin{1}; obj.cols = varargin{2};
            obj.filename = 'temp.dat';
            
            for i = 3:nargin
                if strcmp(varargin{i}, 'filename')
                    obj.filename = varargin{i+1};
                end
                if strcmp(varargin{i}, 'memory')
                    obj.memory_footprint = varargin{i+1};
                end
            end
            
            obj.fid = fopen(obj.filename, 'w');
            frewind(obj.fid);
            
            bytes_needed = obj.rows * obj.cols * 4;
            bytes_used = 0;
            
            % if the memory footprint is not enough to store the whole
            % array, break it into chunks and write zeros to the file.
            if obj.memory_footprint < bytes_needed
                
                while bytes_used < bytes_needed
                
                    bytes_to_write = min(obj.memory_footprint, bytes_needed - bytes_used);
                    
                    A = zeros([bytes_to_write/4 1]);
                    fwrite(obj.fid, A, 'single');
                    
                    disp(['writing ' num2str(bytes_to_write) ' bytes to ' obj.filename ]);
                    
                    bytes_used = bytes_used + bytes_to_write;
                    clear A;
                end
            else
                A = zeros(obj.rows, obj.cols, 'single');   
                fwrite(obj.fid, A, 'float');
                obj.data = A;
            end
                
            fclose(obj.fid);
            
                
        end
        
        %% store rows
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
                seek_bytes = (row_inds(i)-1)*4;
                fseek(obj.fid, seek_bytes, 'bof');
                fwrite(obj.fid, data(i,1), 'single');
                fwrite(obj.fid, data(i,2:size(data,2)), 'single', (obj.cols-1)*4);
            end

            fclose(obj.fid);
           
            
            % if it is part of obj.data, update that as well
            
        end
        
        
        %% store cols
        function storeCols(obj, data, col_inds )
            
            if size(data,2) ~= length(col_inds)
                error('error: the number of columns in data and col_inds do not match');
            end
            
            if size(data,1) ~= obj.rows
                error('error: the number of rows in data does not match the number of rows in the bigmatrix');
            end
            
            
            % store it to the file
            obj.fid = fopen(obj.filename, 'r+');
            fseek(obj.fid, (col_inds(1)-1)*obj.rows*4, 'bof');
            fwrite(obj.fid, data, 'single');
            fclose(obj.fid);
            
            % if it is part of obj.data, update that as well
        end
        
        %% getRows
        function data = getRows(obj, row_inds)
            
            obj.fid = fopen(obj.filename, 'r+');
            
            data = zeros([length(row_inds), obj.cols]);
            
            for i = 1:length(row_inds);
                seek_bytes = (row_inds(i)-1)*4;
                fseek(obj.fid, seek_bytes, 'bof');
                data(i,:) = fread(obj.fid, [1 obj.cols], 'single', (obj.cols-1)*4 );
                %fwrite(obj.fid, data(i,1), 'single');
                %fwrite(obj.fid, data(i,2:size(data,2)), 'single', (obj.cols-1)*4);
            end
            
            fclose(obj.fid);
            
            % store the rows in the current data
            
        end
        
        %% getCols
        function data = getCols(obj, col_inds)
            
            obj.fid = fopen(obj.filename, 'r+');
            fseek(obj.fid, (col_inds(1)-1)*obj.rows*4, 'bof'); 
            data = fread(obj.fid, [obj.rows length(col_inds)], 'float');
            fclose(obj.fid);
            
            % store the cols we have in current data.
        end
    
        
   end %methods
end %classdef
       

% fid = fopen('alphabet.txt', 'r');
% c = fread(fid, '2*char=>char', 2);    % Skip 2 bytes per read

% fid = fopen(B.filename, 'r');
% frewind(fid); fread(fid, [5 5], 'float')
% fseek(fid, 4, 'bof'); fread(fid, [5 1], '*single', 4*4)
% frewind(fid); fread(fid, [5 2], '*single')



           %A = [1 2 3 4 5]' * [1 1 1 1 1];   %  [1 2 3 4 5] * ones(5);
            
%             count = 1;
%             for i= 1:5
%                 for j = 1:5
%                     A(i,j) = count;
%                     count = count + 1;
%                 end
%             end

%             A = zeros(rows, columns, 'single');    
% 
%             obj.data = A;
%             
%             fwrite(obj.fid, A, 'float');
            





% 
% 
%         %% the constructor method
%         function BA = bigarray(varargin)
%             total_rows = varargin{1};
%             num_columns = varargin{2};
%             BA.prefix = 'BIGARRAY_'; BA.path = [pwd '/'];
%             bytesize = 250000000; BA.type = 'matlab .mat file';
%             
%             for i = 3:nargin
%                 if strcmp(varargin{i}, 'filename')
%                     BA.prefix = varargin{i+1};
%                 end
%                 if strcmp(varargin{i}, 'path')
%                     BA.path = varargin{i+1};
%                 end
%                 if strcmp(varargin{i}, 'bytes')
%                     bytesize = varargin{i+1};
%                 end
%                 if strcmp(varargin{i}, 'type')
%                     BA.type = varargin{i+1};
%                 end
%             end
%             
%             BA.bigarray_size = [total_rows num_columns];
%             rows_per_block = round(bytesize / (num_columns*8));
%             BA.block_size = [rows_per_block num_columns];
%             BA.num_blocks = ceil(total_rows/rows_per_block);
%             
%             if BA.num_blocks == 1
%                 if strcmp(BA.type, 'memmapfile')
%                     BA.filenames{1} = [BA.path BA.prefix num2str(1) '.dat'];
%                 else 
%                     BA.filenames{1} = [BA.path BA.prefix num2str(1) '.mat'];
%                 end
%                 BA.row_bounds{1} = [1 total_rows];
%                 BA.allocate(BA.filenames{1});
%             else
%                 for i = 1:BA.num_blocks
%                     if i ~= BA.num_blocks
%                         if strcmp(BA.type, 'memmapfile')
%                             BA.filenames{i} = [BA.path BA.prefix num2str(i) '.dat'];
%                         else 
%                             BA.filenames{i} = [BA.path BA.prefix num2str(i) '.mat'];
%                         end
%                         BA.row_bounds{i} = [1 BA.block_size(1)] + BA.block_size(1)*(i-1);
%                         BA.allocate(BA.filenames{i});
%                     else
%                         if strcmp(BA.type, 'memmapfile')
%                             BA.filenames{i} = [BA.path BA.prefix num2str(i) '.dat'];
%                         else 
%                             BA.filenames{i} = [BA.path BA.prefix num2str(i) '.mat'];
%                         end
%                         previous_row = (i-1)*BA.block_size(1);
%                         BA.row_bounds{i} = [1 total_rows - previous_row] + BA.block_size(1)*(i-1);
%                         BA.allocate(BA.filenames{i});
%                     end
%                 end
%             end
%             if strcmp(BA.type, 'memmapfile')
%                 BA.memmap = memmapfile(BA.filenames{1}, 'Offset', 0, 'Writable', true,      ...    
%                 'Format', {'double' BA.block_size 'x'});
%                 BA.block_data = 'unused - needed for matlab .mat type'; BA.loaded_block = 'unused - needed for matlab .mat type';
%             else
%                 load(BA.filenames{1}); BA.block_data = D;  clear D;     
%                 BA.loaded_block = 1;
%                 BA.memmap = 'unused - needed for memmapfile type';
%             end
%         end
%        
%         %% the get_rows method
%         function mat1 = get_rows(varargin)  % (BA, row_lims)
%             BA = varargin{1};  row_lims = varargin{2}; nosave = 0;
%             if nargin == 3; if strcmp('nosave', varargin{3}); nosave = 1; end; end
%             mat1 = []; 
%             
%             % if only 1 row is requested
%             if (row_lims(1) == row_lims(2)) || max(size(row_lims)) == 1
%                 block = ceil(row_lims(1)/BA.block_size(1));
%                 r = row_lims(1) - (block-1)*BA.block_size(1);
%                 if strcmp(BA.type, 'memmapfile')
%                     if ~strcmp(BA.memmap.filename, BA.filenames{block})
%                         BA.memmap.filename = BA.filenames{block};
%                     end
%                     mat1 = BA.memmap.data.x(r,:);
%                 else
%                     if BA.loaded_block ~= block
%                         if ~nosave
%                             D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D;
%                         end
%                         load(BA.filenames{block}); BA.block_data = D;  clear D;
%                         BA.loaded_block = block;
%                     end
%                     mat1 = BA.block_data(r,:);
%                 end
%                 
%             % if multiple rows are requested
%             else
%                 firstblock = ceil(row_lims(1)/BA.block_size(1));
%                 lastblock = ceil(row_lims(2)/BA.block_size(1));
%                 for i = firstblock:lastblock
%                     if (i == firstblock) && (i == lastblock)
%                         r1 = row_lims(1) - (i-1)*BA.block_size(1);
%                         r2 = row_lims(2) - (i-1)*BA.block_size(1);
%                     elseif (i == firstblock)
%                         r1 = row_lims(1) - (i-1)*BA.block_size(1);
%                         r2 = BA.block_size(1);
%                     elseif (i == lastblock)
%                         r1 = 1;
%                         r2 = row_lims(2) - (i-1)*BA.block_size(1);
%                     else
%                         r1 = 1;
%                         r2 = BA.block_size(1);
%                     end
% 
%                     if strcmp(BA.type, 'memmapfile')
%                         if ~strcmp(BA.memmap.filename, BA.filenames{i})
%                             BA.memmap.filename = BA.filenames{i};
%                         end
%                         mat1 = [mat1; BA.memmap.data.x(r1:r2,:)];
%                     else
%                         if BA.loaded_block ~= i
%                             if ~nosave
%                                 D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D;
%                             end
%                             load(BA.filenames{i}); BA.block_data = D;  clear D;
%                             BA.loaded_block = i;
%                         end
%                         mat1 = [mat1; BA.block_data(r1:r2,:)];
%                     end
%                 end
%             end
%         end
% 
%         %% the store_rows method
%         function store_rows(BA, mat, row_lims)
% 
%             firstblock = ceil(row_lims(1)/BA.block_size(1));
%             lastblock = ceil(row_lims(2)/BA.block_size(1));
%             mat_top_row = 1;
%             
%             for i = firstblock:lastblock
%                 if (i == firstblock) && (i == lastblock)
%                     r1 = row_lims(1) - (i-1)*BA.block_size(1);
%                     r2 = row_lims(2) - (i-1)*BA.block_size(1);
%                 elseif (i == firstblock)
%                     r1 = row_lims(1) - (i-1)*BA.block_size(1);
%                     r2 = BA.block_size(1);
%                 elseif (i == lastblock)
%                     r1 = 1;
%                     r2 = row_lims(2) - (i-1)*BA.block_size(1);
%                 else
%                     r1 = 1;
%                     r2 = BA.block_size(1);
%                 end
%                 if strcmp(BA.type, 'memmapfile')
%                     if ~strcmp(BA.memmap.filename, BA.filenames{i})
%                         BA.memmap.filename = BA.filenames{i};
%                     end
%                     BA.memmap.data.x(r1:r2,:) = mat(mat_top_row:mat_top_row+r2-r1,:);
%                     mat_top_row = mat_top_row + r2 -r1 + 1;
%                 else
%                     if BA.loaded_block ~= i
%                         D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D;
%                         load(BA.filenames{i}); BA.block_data = D;  clear D;
%                         BA.loaded_block = i;
%                     end
%                     BA.block_data(r1:r2,:) = mat(mat_top_row:mat_top_row+r2-r1,:);
%                     mat_top_row = mat_top_row + r2 -r1 + 1;
%                 end
%             end
%         end
%         
%         %% the force_save method
%         function force_save(BA,a) %#ok<INUSD>
%             if ~strcmp(BA.type, 'memmapfile')
%                 D = BA.block_data; save(BA.filenames{BA.loaded_block},'D'); clear D; %#ok<NASGU>
%             end
%         end
%         
%         %% the cleanup method
%         function cleanup(BA)
%             if strcmp(BA.type, 'memmapfile')
%                 for i = 1:length(BA.filenames)
%                     cmd = ['rm ' BA.filenames{i}];
%                     system(cmd);
%                 end
%             else
%                 for i = 1:length(BA.filenames)
%                     cmd = ['rm ' BA.filenames{i}];
%                     system(cmd);
%                 end
%             end
%         end
%         
%         %% the allocate method
%         function allocate(BA, filename)
%             if strcmp(BA.type, 'memmapfile')
%                 ELEMENTS_NEEDED = prod(BA.block_size);
%                 BYTES = ELEMENTS_NEEDED*8;
%                 if exist(filename, 'file')
%                     d = dir(filename);
%                     if BYTES == d.bytes
%                         disp(['...' filename ' already exists and is correct size!']);
%                         return
%                     end
%                 end
%                 disp(['... allocating memory-mapped file ' filename ]);
%                 fid = fopen(filename,'w+');
%                 block = 1000;
%                 for i = 1:BA.block_size(1)
%                     if mod(i,block) ==0
%                         fwrite(fid, zeros([block BA.block_size(2)]), 'double'); 
%                         ELEMENTS_NEEDED = ELEMENTS_NEEDED - block*BA.block_size(2);
%                     end
%                 end
%                 fwrite(fid, zeros([ELEMENTS_NEEDED 1]), 'double'); 
%                 fclose(fid);
%             else
%                 disp(['... allocating matlab .mat file ' filename ]);
%                 D = zeros(BA.block_size); %#ok<NASGU>
%                 save(filename, 'D');
%             end
%         end
%         
%    end% methods
% end% classdef