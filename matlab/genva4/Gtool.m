function varargout = Gtool(varargin)
% GTOOL MATLAB code for Gtool.fig
%      GTOOL, by itself, creates a new GTOOL or raises the existing
%      singleton*.
%
%      H = GTOOL returns the handle to a new GTOOL or the handle to
%      the existing singleton*.
%
%      GTOOL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GTOOL.M with the given input arguments.
%
%      GTOOL('Property','Value',...) creates a new GTOOL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Gtool_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Gtool_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Gtool

% Last Modified by GUIDE v2.5 19-May-2011 10:54:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Gtool_OpeningFcn, ...
                   'gui_OutputFcn',  @Gtool_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Gtool is made visible.
function Gtool_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Gtool (see VARARGIN)

% Choose default command line output for Gtool
handles.output = hObject;
setpaths();
warning off;

handles.hFig = figure('Toolbar','none','Menubar','none','NumberTitle', 'off', 'Name', 'No data loaded');
handles.hIm = imshow(zeros(800)); 
handles.hSP = imscrollpanel(handles.hFig,handles.hIm);
handles.hFigAxis = gca;
handles.api = iptgetapi(handles.hSP);
handles.hStats = figure('Toolbar','none','Menubar','none','NumberTitle', 'off', 'Name', 'No data loaded');
close(handles.hStats);

handles = clearAnnotations(handles);

handles.t = 1;

% start matlabpool for parallel processing
if matlabpool('size') > 0
    matlabpool close;
end
matlabpool

%% ========================== DEFAULT PARAMETERS ==========================

handles.folder = '/home/ksmith/data/Sinergia/Geneva/Laurence/2010-09-16/RMS04/RMS04TTX/';
if ~exist(handles.folder, 'dir')
    handles.folder = pwd;
end
%handles.CLim = [0 255];
handles.CLim = [0 65535];

%% ========================================================================

load('colors.mat');
handles.colors = colors;

set(handles.zslider, 'Enable', 'inactive');
set(handles.zslider, 'BackgroundColor', [.5 .5 .5]);
set(handles.zslider, 'Visible', 'off');

set(handles.hSP,'Units','normalized','Position',[0 0 1 1])
set(handles.hFig,'KeyPressFcn',{@keypresshandler, handles}); % set an event listener for keypresses
handles.ovpanel = imoverviewpanel(handles.ovpanel, handles.hIm);

set(handles.visButtongroup,'SelectionChangeFcn',@visButtongroup_SelectionChangeFcn);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Gtool wait for user response (see UIRESUME)
% uiwait(handles.Gtool);


% --- Outputs from this function are returned to the command line.
function varargout = Gtool_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on slider movement.
function zslider_Callback(hObject, eventdata, handles)
% hObject    handle to zslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

v = get(hObject,'Value');
set(hObject, 'Value', round(v));

handles = updateImage(handles);

%get(hObject,'Value')

% --- Executes during object creation, after setting all properties.
function zslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function tslider_Callback(hObject, eventdata, handles)
% hObject    handle to tslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

v = get(hObject,'Value');
set(hObject, 'Value', round(v));
handles.t = get(hObject, 'Value');

handles = updateImage(handles);

if ~isempty(handles.D(1).Area)
    handles = updateStats(handles);
    guidata(hObject, handles);
end

%get(hObject,'Value')
guidata(hObject, handles);



% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function tslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in loadbutton.
function loadbutton_Callback(hObject, eventdata, handles)
% hObject    handle to loadbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = openFolder(handles);
guidata(hObject, handles);

% --- Executes on button press in savebutton.
function savebutton_Callback(hObject, eventdata, handles)
% hObject    handle to savebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

folder = uigetdir(handles.folder, 'Select folder to save the TIF sequence');

if folder ~= 0
    for t = 1:handles.tmax
        [pth name ext] = fileparts(handles.filenames{t});
        filename = [folder '/' name ext];
        disp(['writing ' filename]);
        writeMultiPageTiff(handles.Data{t}, filename);
    end
else
    disp('no folder selected');
end

guidata(hObject, handles);

% --- Executes on button press in zoomin.
function zoomin_Callback(hObject, eventdata, handles)
% hObject    handle to zoomin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

mag = handles.api.getMagnification();
mag = mag * 1.5;
handles = zoomFun(handles, mag);
guidata(hObject, handles);

% --- Executes on button press in zoom1.
function zoom1_Callback(hObject, eventdata, handles)
% hObject    handle to zoom1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

mag = handles.api.getMagnification();
mag = 1;
handles = zoomFun(handles, mag);
guidata(hObject, handles);

% --- Executes on button press in zoomout.
function zoomout_Callback(hObject, eventdata, handles)
% hObject    handle to zoomout (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

mag = handles.api.getMagnification();
mag = mag / 1.5;
handles = zoomFun(handles, mag);
guidata(hObject, handles);


function handles = zoomFun(handles, mag)

handles.api.setMagnification(mag);



function handles = openFolder(handles)


if isfield(handles, 'Data')
    choice = questdlg('Loading new data will clear the annotations. Continue?', 'Load New Data',...
    'Load New Data', 'Cancel', 'Cancel');
else
    choice = '';
end


switch choice
    case 'Cancel'
        return;
    otherwise
        handles = clearAnnotations(handles);

        folder = uigetdir(handles.folder, 'Select folder containing TIF sequence');

        if folder ~= 0

            prompt={'Sample size in x-direction (\mu m):', 'Sample size in y-direction (\mu m):', 'Sample size in z-direction (\mu m):'};
            name = 'Stack dimensions';
            numlines = 1;
            defaultanswer = {'200', '200', '30'};
            options.Resize='on';
            options.WindowStyle='normal';
            options.Interpreter='tex';
            answer=inputdlg(prompt,name,numlines,defaultanswer,options);
            d =[];
            
            if isempty(answer)
                dx = 1;
                dy = 1;
                dz = 1;
            else
                dx = str2double(answer{1});
                dy = str2double(answer{2});
                dz = str2double(answer{3});
                d = [dx dy dz];
            end
            if isempty(d)
                disp('Cancelled');
                %keyboard;
                return;
            end
            if ~isa(d, 'double')
                disp('Cancelled');
                %keyboard;
                return;
            end
            pause(0.1);

            handles.folder = [folder '/'];
            handles = loadData(handles);

            handles.spacing(1) = dy / size(handles.Data{1},1);
            handles.spacing(2) = dx / size(handles.Data{1},2);
            handles.spacing(3) = dz / size(handles.Data{1},3);

            disp([ '[' num2str(handles.spacing) '] um / voxel']);
        else
            disp('no folder selected');
        end
end




function handles = loadData(handles)

% clear the old data
handles.Data = {};
handles.DataMax = [];
handles.I = [];
handles.ProjMIP = {};
handles.ProjSTD = {};
handles.t = 0;
handles.tmax = 0;
handles.zmax = 0;
handles.spacing = [];

pat = '*.tif';
d = dir([handles.folder pat]);
handles.DataMax = 0;
handles.tmax = length(d);
handles.dlist = cell(1, handles.tmax);
handles.glist = cell(1, handles.tmax);

for t = 1:handles.tmax  
    handles.filenames{t} = [handles.folder d(t).name];

    % get the raw data
    disp(['loading [' num2str(t) '/' num2str(handles.tmax) '] ' handles.filenames{t}]);
    handles.Data{t} = readMultiPageTiff(handles.filenames{t});
    handles.DataMax = max(handles.DataMax, single(max(handles.Data{t}(:) ) ) );
end

% set t slider
set(handles.tslider, 'Value', 1, 'min', 1, 'max', handles.tmax, 'sliderstep', [1/handles.tmax 1/handles.tmax]);

% set z slider
handles.zmax = size(handles.Data{1},3);
set(handles.zslider, 'Value', 1, 'min', 1, 'max', handles.zmax, 'sliderstep', [1/handles.zmax 1/handles.zmax]);

% compute projections
handles = miProj(handles, 1:handles.tmax);
handles = stdProj(handles, handles.Data, 1:handles.tmax);
handles.t = 1;

if get(handles.pushInvertContrast, 'Value')
    CLim = [65535-handles.DataMax  65535];
else
 	CLim = [0 handles.DataMax];
end
handles.CLim = CLim;

if isvalid(handles.hMeasure)
    delete(handles.hMeasure);
end

set(handles.pushMeasure, 'Value', 0);


% set the image to be the first std projection
handles = updateImage(handles);
handles = updateIDList(handles);
handles = updateStats(handles);




%% ================== UPDATE IMAGE ===================================


function handles = updateImage(handles)


% set the filename at top of the figure
[pathstr name ext] = fileparts(handles.filenames{handles.t});

% get image if it is z slice view or max intensity proj or std int proj
if get(handles.radSlice, 'Value')
    z = get(handles.zslider, 'Value');
    handles.I = handles.Data{handles.t}(:,:,z);
    set(handles.hFig, 'Name', [name ext ' [z=' num2str(z) ']' ]);
elseif get(handles.radMax, 'Value')
  	handles.I = handles.ProjMIP{handles.t};
    set(handles.hFig, 'Name', [name ext]);
elseif get(handles.radSTD, 'Value')
    handles.I = handles.ProjSTD{handles.t};
    set(handles.hFig, 'Name', [name ext]);
else
    error('invalid view mode');
end

% invert the contrast if necessary
if get(handles.pushInvertContrast, 'Value')
    %handles.I = invert8bitImage(handles.I);
    handles.I = invert16bitImage(handles.I);
end

handles.RawI = handles.I;

if get(handles.checkAnn, 'Value')

    Ir = mat2gray(handles.I, double(handles.CLim)); 
    Ig = Ir; Ib = Ir;
    
    if get(handles.radSlice, 'Value')
            
        % Nuclei: z-slice mode show each volume slice
        dlist = handles.dlist{handles.t};
        for d = dlist
            if handles.D(d).ID ~= 0
                color = handles.colors(handles.D(d).ID,:);
            else
                % non-annotated color
                if get(handles.pushInvertContrast, 'Value')
                    color = [0 0 0];
                else
                    color = [1 1 1];
                end
            end
            gray = rgb2gray(color);
            cfactor = [1 1 1] + (color - gray) * .5;
            cfactor2 = [1 1 1] + (color - gray) * .1;
            hsv = rgb2hsv(color);
            hsv(2) = hsv(2)*.5; hsv(3) = hsv(3) * .75;
            cdim = hsv2rgb(hsv);
            
            if handles.D(d).ID == get(handles.listID, 'Value');
                z = get(handles.zslider,'Value');
                Ir(handles.D(d).pslices{z}) = color(1);
                Ig(handles.D(d).pslices{z}) = color(2);
                Ib(handles.D(d).pslices{z}) = color(3);
                Ir(handles.D(d).slices{z}) = Ir(handles.D(d).slices{z}) *cfactor(1);
                Ig(handles.D(d).slices{z}) = Ig(handles.D(d).slices{z}) *cfactor(2);
                Ib(handles.D(d).slices{z}) = Ib(handles.D(d).slices{z}) *cfactor(3);
            else
                z = get(handles.zslider,'Value');
                Ir(handles.D(d).pslices{z}) = cdim(1);
                Ig(handles.D(d).pslices{z}) = cdim(2);
                Ib(handles.D(d).pslices{z}) = cdim(3);
                Ir(handles.D(d).slices{z}) = Ir(handles.D(d).slices{z}) *cfactor2(1);
                Ig(handles.D(d).slices{z}) = Ig(handles.D(d).slices{z}) *cfactor2(2);
                Ib(handles.D(d).slices{z}) = Ib(handles.D(d).slices{z}) *cfactor2(3);
            end
        end
        
        % Growth Cones: z-slice mode show each volume slice
        glist = handles.glist{handles.t};
        for g = glist
            if handles.G(g).ID ~= 0
                color = handles.colors(handles.G(g).ID,:);
            else
                % non-annotated color

                if get(handles.pushInvertContrast, 'Value')
                    color = [0 0 0];
                else
                    color = [1 1 1];
                end
            end
            gray = rgb2gray(color);
            cfactor = [1 1 1] + (color - gray) * .5;
            cfactor2 = [1 1 1] + (color - gray) * .1;
            hsv = rgb2hsv(color);
            hsv(2) = hsv(2)*.5; hsv(3) = hsv(3) * .75;
            cdim = hsv2rgb(hsv);
            
            if handles.G(g).ID == get(handles.listID, 'Value');
                z = get(handles.zslider,'Value');
                p = handles.G(g).pslices{z};
                p = p(randperm(length(p)));
                p1 = p(1:2:end);
                Ir(p1) = color(1);
                Ig(p1) = color(2);
                Ib(p1) = color(3);
%                 Ir(handles.G(g).pslices{z}) = color(1);
%                 Ig(handles.G(g).pslices{z}) = color(2);
%                 Ib(handles.G(g).pslices{z}) = color(3);
                Ir(handles.G(g).slices{z}) = Ir(handles.G(g).slices{z}) *cfactor(1);
                Ig(handles.G(g).slices{z}) = Ig(handles.G(g).slices{z}) *cfactor(2);
                Ib(handles.G(g).slices{z}) = Ib(handles.G(g).slices{z}) *cfactor(3);
            else
                z = get(handles.zslider,'Value');
                p = handles.G(g).pslices{z};
                p = p(randperm(length(p)));
                p1 = p(1:2:end);
                Ir(p1) = cdim(1);
                Ig(p1) = cdim(2);
                Ib(p1) = cdim(3);
%                 Ir(handles.G(g).pslices{z}) = cdim(1);
%                 Ig(handles.G(g).pslices{z}) = cdim(2);
%                 Ib(handles.G(g).pslices{z}) = cdim(3);
                Ir(handles.G(g).slices{z}) = Ir(handles.G(g).slices{z}) *cfactor2(1);
                Ig(handles.G(g).slices{z}) = Ig(handles.G(g).slices{z}) *cfactor2(2);
                Ib(handles.G(g).slices{z}) = Ib(handles.G(g).slices{z}) *cfactor2(3);
            end
        end
    else
        
        % Nuclei: projection modes, show only projection
        dlist = handles.dlist{handles.t};
        for d = dlist
            if handles.D(d).ID ~= 0
                color = handles.colors(handles.D(d).ID,:);
            else
                % non-annotated color
                if get(handles.pushInvertContrast, 'Value')
                    color = [0 0 0];
                else
                    color = [1 1 1];
                end
            end
            gray = rgb2gray(color);
            cfactor = [1 1 1] + (color - gray) * .5;
            cfactor2 = [1 1 1] + (color - gray) * .1;
            hsv = rgb2hsv(color);
            hsv(2) = hsv(2)*.5; hsv(3) = hsv(3) * .75;
            cdim = hsv2rgb(hsv);
            
            if handles.D(d).ID == get(handles.listID, 'Value');
                Ir(handles.D(d).PerimList) = color(1);
                Ig(handles.D(d).PerimList) = color(2);
                Ib(handles.D(d).PerimList) = color(3);
                Ir(handles.D(d).PixelIdxList) = Ir(handles.D(d).PixelIdxList)*cfactor(1);
                Ig(handles.D(d).PixelIdxList) = Ig(handles.D(d).PixelIdxList)*cfactor(2);
                Ib(handles.D(d).PixelIdxList) = Ib(handles.D(d).PixelIdxList)*cfactor(3);
            else
                Ir(handles.D(d).PerimList) = cdim(1);
                Ig(handles.D(d).PerimList) = cdim(2);
                Ib(handles.D(d).PerimList) = cdim(3);
                Ir(handles.D(d).PixelIdxList) = Ir(handles.D(d).PixelIdxList)*cfactor2(1);
                Ig(handles.D(d).PixelIdxList) = Ig(handles.D(d).PixelIdxList)*cfactor2(2);
                Ib(handles.D(d).PixelIdxList) = Ib(handles.D(d).PixelIdxList)*cfactor2(3);
            end
        end
        
        % Growth Cone: projection modes, show only projection
        glist = handles.glist{handles.t};
        for g = glist
            if handles.G(g).ID ~= 0
                color = handles.colors(handles.G(g).ID,:);
            else
                % non-annotated color
                if get(handles.pushInvertContrast, 'Value')
                    color = [0 0 0];
                else
                    color = [1 1 1];
                end
            end
            gray = rgb2gray(color);
            cfactor = [1 1 1] + (color - gray) * .5;
            cfactor2 = [1 1 1] + (color - gray) * .1;
            hsv = rgb2hsv(color);
            hsv(2) = hsv(2)*.5; hsv(3) = hsv(3) * .75;
            cdim = hsv2rgb(hsv);
            
            if handles.G(g).ID == get(handles.listID, 'Value');
                p = handles.G(g).PerimList;
                p = p(randperm(length(p)));
                p1 = p(1:2:end);
                %p2 = p(2:2:end);
                Ir(p1) = color(1);
                Ig(p1) = color(2);
                Ib(p1) = color(3);
%                 Ir(p2) = .5;
%                 Ig(p2) = 1;
%                 Ib(p2) = .5;
%                 Ir(handles.G(g).PerimList(1:end)) = color(1);
%                 Ig(handles.G(g).PerimList(1:end)) = color(2);
%                 Ib(handles.G(g).PerimList(1:end)) = color(3);
                Ir(handles.G(g).PixelIdxList) = Ir(handles.G(g).PixelIdxList)*cfactor(1);
                Ig(handles.G(g).PixelIdxList) = Ig(handles.G(g).PixelIdxList)*cfactor(2);
                Ib(handles.G(g).PixelIdxList) = Ib(handles.G(g).PixelIdxList)*cfactor(3);
            else
                p = handles.G(g).PerimList;
                p = p(randperm(length(p)));
                p1 = p(1:2:end);
                Ir(p1) = cdim(1);
                Ig(p1) = cdim(2);
                Ib(p1) = cdim(3);
%                 Ir(handles.G(g).PerimList(1:end)) = cdim(1);
%                 Ig(handles.G(g).PerimList(1:end)) = cdim(2);
%                 Ib(handles.G(g).PerimList(1:end)) = cdim(3);
                Ir(handles.G(g).PixelIdxList) = Ir(handles.G(g).PixelIdxList)*cfactor2(1);
                Ig(handles.G(g).PixelIdxList) = Ig(handles.G(g).PixelIdxList)*cfactor2(2);
                Ib(handles.G(g).PixelIdxList) = Ib(handles.G(g).PixelIdxList)*cfactor2(3);
            end
        end
    end
    I(:,:,1) = Ir;
    I(:,:,2) = Ig;
    I(:,:,3) = Ib;

    handles.I = I;
end
    
    
handles.api.replaceImage(handles.I, 'PreserveView', 1);
set(handles.hFigAxis, 'CLim', handles.CLim);



%% =================================================================












function I = invert8bitImage(I)
    I = single(I);
    I = uint8(abs(255-I));

    
function I = invert16bitImage(I)
    I = single(I);
    I = uint16(abs(65535-I));
    

function handles = miProj(handles, T)
for t = T
    I = max(handles.Data{t}, [], 3);
    handles.ProjMIP{t} = I;  
    if length(T) ~= 1
        fprintf('|');
    end
end
fprintf('\n');





function handles = stdProj(handles, Data, T)


if length(T) > 1
    tic;
    disp('Computing Standard Deviation Projection, please wait...');
end

if get(handles.pushInvertContrast, 'Value')
    ca =  65535 - handles.CLim(2);
    cb =  65535 - handles.CLim(1);
    ca = double(ca);
    cb = double(cb);
else
    ca = double(handles.CLim(1));
    cb = double(handles.CLim(2));
end

parfor t = T
    I = std(single(Data{t}), [], 3);
    I = mat2gray(I);
    I = imadjust(I, [0 1], [ca/65535 cb/65535]);
    im{t} = uint16(65535 * I);
end

for t = T
    handles.ProjSTD{t} = im{t};
end

if length(T) > 1
    toc;
end
  
  

function visButtongroup_SelectionChangeFcn(hObject, eventdata)

handles = guidata(hObject); 
 
switch get(eventdata.NewValue,'Tag')
    case 'radMax'
        set(handles.zslider, 'Enable', 'inactive');
        set(handles.zslider, 'BackgroundColor', [.5 .5 .5]);
        set(handles.zslider, 'Visible', 'off');
        %handles = miProj(handles, handles.t);
        handles = updateImage(handles);
        %handles = miProj(handles, 1:handles.tmax);
    case 'radSTD'
        set(handles.zslider, 'Enable', 'inactive');
        set(handles.zslider, 'BackgroundColor', [.5 .5 .5]);
        set(handles.zslider, 'Visible', 'off');
        %handles = stdProj(handles, handles.Data, handles.t);
        handles = updateImage(handles);
        %handles = stdProj(handles, handles.Data, 1:handles.tmax);
    case 'radSlice'
        set(handles.zslider, 'Enable', 'on');
        set(handles.zslider, 'BackgroundColor', [.9 .9 .9]);
        set(handles.zslider, 'Visible', 'on');
        handles = updateImage(handles);
    otherwise
    
end
guidata(hObject, handles);



function setpaths()
addpath([pwd '/src/']);
addpath([pwd '/src/bm3d/']);
addpath([pwd '/src/segmentation/']);
addpath([pwd '/src/vol3d/']);


% --- Executes on button press in debug.
function debug_Callback(hObject, eventdata, handles)
% hObject    handle to debug (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

keyboard;
guidata(hObject, handles);

% --- Executes on button press in pushContrast.
function pushContrast_Callback(hObject, eventdata, handles)
% hObject    handle to pushContrast (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%imcontrast(handles.hSP);

CLim = get(handles.hFigAxis, 'CLim');
handles.CLim = CLim;
guidata(hObject, handles);


% --- Executes on button press in pushAdjContrast.
function pushAdjContrast_Callback(hObject, eventdata, handles)
% hObject    handle to pushAdjContrast (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


set(handles.checkAnn, 'Value', 0);
handles.api.replaceImage(handles.RawI, 'PreserveView', 1);
imcontrast(handles.hSP);
%handles.api.replaceImage(handles.I, 'PreserveView', 1);

% --- Executes on button press in pushResetContrast.
function pushResetContrast_Callback(hObject, eventdata, handles)
% hObject    handle to pushResetContrast (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% set reasonable contrast limits
if get(handles.pushInvertContrast, 'Value')
    CLim = [65535-handles.DataMax  65535];
else
 	CLim = [0 handles.DataMax];
end
handles.CLim = CLim;
handles = updateImage(handles);
guidata(hObject, handles);


% --- Executes on button press in pushInvertContrast.
function pushInvertContrast_Callback(hObject, eventdata, handles)
% hObject    handle to pushInvertContrast (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

old = handles.CLim;
if get(handles.pushInvertContrast, 'Value')
    handles.CLim(1) = 65535 - old(2);
    handles.CLim(2) = 65535 - old(1);
else
   	handles.CLim(2) = 65535 - old(1);
  	handles.CLim(1) = abs(65535 - old(2));
end

handles = updateImage(handles);
guidata(hObject, handles);


% --- Executes on button press in pushDenoise.
function pushDenoise_Callback(hObject, eventdata, handles)
% hObject    handle to pushDenoise (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


choice = questdlg('What would you like to denoise? It may take a few minutes...', 'Denoising',...
    'All stacks', 'Current stack', 'Cancel', 'Cancel');

switch choice
    case 'Current stack'
        prompt={'Enter the denoising strength between .5 and 20 (default = 3)'};
        name = 'Denoising sigma';
        numlines = 1;
        defaultanswer = {'3.0'};
        options.Resize='on';
        options.WindowStyle='normal';
        options.Interpreter='tex';
        answer=inputdlg(prompt,name,numlines,defaultanswer,options);
        BM3Dsigma = str2double(answer{1});
        if ~isa(BM3Dsigma, 'double')
            return;
        end
        pause(0.1);
        disp('DENOISING, please wait...');
        tic;
        handles.Data{handles.t} = denoiseBM3D(handles.Data{handles.t}, BM3Dsigma);
        handles = miProj(handles, handles.t);
        handles = stdProj(handles, handles.Data, handles.t);
        handles = updateImage(handles);
        guidata(hObject, handles);
        toc;
    case 'All stacks'
        prompt={'Enter the denoising strength between .5 and 20 (default = 3)'};
        name = 'Denoising sigma';
        numlines = 1;
        defaultanswer = {'3.0'};
        options.Resize='on';
        options.WindowStyle='normal';
        options.Interpreter='tex';
        answer=inputdlg(prompt,name,numlines,defaultanswer,options);
        BM3Dsigma = str2double(answer{1});
        if ~isa(BM3Dsigma, 'double')
            return;
        end
        pause(0.1);
        disp('DENOISING the entire series, please wait...');
        tic;
        for t = 1:handles.tmax 
            disp(['[' num2str(t) '/' num2str(handles.tmax) ']' ]);
            handles.Data{t} = denoiseBM3D(handles.Data{t}, BM3Dsigma);
        end
        handles = miProj(handles, handles.t);
        handles = stdProj(handles, handles.Data, 1:handles.tmax);
        handles = updateImage(handles);
        guidata(hObject, handles);
        toc;
    case 'Cancel'
        
end



% --- Executes during object deletion, before destroying properties.
function Gtool_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to Gtool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes when user attempts to close Gtool.
function Gtool_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to Gtool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
delete(hObject);







function [zsmooth handles] = smoothZselection(zdata, handles)

zsmooth = smooth(zdata, 9, 'loess');

% figure(99);
% plot(zdata, 1:handles.zmax, 'b--'); 
% hold on;
% plot( zsmooth, 1:handles.zmax, 'r');
% hold off;
% %axis([1 handles.zmax 0 max(zsmooth)]);










% --- Executes on button press in pushObject.
function pushObject_Callback(hObject, eventdata, handles)
% hObject    handle to pushObject (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


if isvalid(handles.ellipse)
    delete(handles.ellipse);
end
    
visrect = handles.api.getVisibleImageRect();
r = round(visrect(2) + visrect(4)/2);
c = round(visrect(1) + visrect(3)/2);

diameter = str2double(get(handles.editDiameter, 'String'));

handles.ellipse = imellipse(handles.hFigAxis, [c-(diameter/2) r-(diameter/2) diameter diameter ]);
addNewPositionCallback(handles.ellipse,@(p) title(mat2str(p,3)));
fcn = makeConstrainToRectFcn('imellipse',get(handles.hFigAxis,'XLim'),get(handles.hFigAxis,'YLim'));
setPositionConstraintFcn(handles.ellipse,fcn);   

setFixedAspectRatioMode(handles.ellipse, 1);
currentID = handles.IDs(get(handles.listID, 'Value'));
color = handles.colors(currentID,:);
setColor(handles.ellipse, color);

guidata(hObject, handles);





% --- Executes on button press in pushSegment.
function pushSegment_Callback(hObject, eventdata, handles)
% hObject    handle to pushSegment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isvalid(handles.ellipse)
    disp('Create a new object to segment');
    return;
end

position = handles.ellipse.getPosition();
position = double(position);

rmin = position(2);
cmin = position(1);
rmax = position(2) + position(4);
cmax = position(1) + position(3);

rcen = round(position(2) + position(4)/2);
ccen = round(position(1) + position(3)/2);

rcen2 = round(position(4)/2);
ccen2 = round(position(3)/2);

% get the z-position
if ~get(handles.radSlice, 'Value')
    zdata = [squeeze(handles.Data{handles.t}(rmin:rmax,cmin:cmax,:))];
    zdata = double(zdata);
    zdata = mean(zdata,1);
    zdata = mean(zdata,2);
    zdata = squeeze(zdata);
    [zdata handles] = smoothZselection(zdata, handles);
    [maxi zcen2] = max(zdata);
else
    zcen2 = get(handles.zslider,'Value');
end
    
ImageSpacing = handles.spacing / handles.spacing(1);
disp(['spacing = [' num2str(ImageSpacing) ']']);


rawdata = double(handles.Data{handles.t}(rmin:rmax, cmin:cmax, :));
seedPoint = [rcen2 ccen2 zcen2];
radius = ceil(position(4)/2);

%% sigmoid param
SigWeight = get(handles.editSigmoid, 'String');
SigWeight = str2num(SigWeight);


if isvalid(handles.ellipse)
    delete(handles.ellipse);
end

tic
segmentation = SGAC(rawdata, ImageSpacing, seedPoint, radius, SigWeight);
toc

offsets = double([rmin rmax cmin cmax]);
[Dnew handles] = extractNucleiSegmentation(rawdata,segmentation, [size(handles.Data{handles.t},1) size(handles.Data{handles.t},2)], offsets, handles);

if ~isempty(Dnew)
    dlength = length(handles.D);
    if (dlength == 1) && isempty(handles.D.Area)
        Dnew.zCenter = zcen2;
        handles.D = Dnew;
    else
        Dnew.zCenter = zcen2;
        handles.D(dlength+1) = Dnew;
    end

    handles.dlist{handles.t}(end+1) = length(handles.D);
end

handles = updateImage(handles);
handles = updateStats(handles);
guidata(hObject, handles);




function [D handles] = extractNucleiSegmentation(rawdata,V, IMSIZE, offsets, handles)

D = [];

I = zeros(IMSIZE);
Vflat = max(V,[],3);

rmin = offsets(1); rmax = offsets(2); cmin = offsets(3); cmax = offsets(4);
I(rmin:rmax, cmin:cmax) = Vflat;

L = bwlabel(I); 
p = regionprops(L, 'Area', 'Eccentricity', 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');

if isempty(p)
    disp('Segmentation was empty!');
    return;
end

arealist = [p.Area];
[area,bigind] = max(arealist);
p = p(bigind);

I = zeros(IMSIZE);
I(p.PixelIdxList) = 1;
perim = bwperim(I);
p.PerimList = find(perim);


% compute interesting statistics
Vbin = V > 0;
p.Volume = sum(Vbin(:));
Ivec = rawdata(Vbin);
p.meanIntensity = mean(Ivec(:));
p.stdIntensity = std(Ivec(:));

%keyboard;

p.t = handles.t;
p.ID = 0;

p.slices = cell(1,handles.zmax);
for z = 1:handles.zmax
    I = zeros(IMSIZE);
    I(rmin:rmax, cmin:cmax) = V(:,:,z);
    p.slices{z} = find(I);   
    p.pslices{z} = find(bwperim(I));
end


% check for an ID conflict with the new annotation
newID = handles.IDs(get(handles.listID, 'Value'));
conflict = 0;
dlist = handles.dlist{handles.t};
for d = dlist
    if handles.D(d).ID == newID
        conflict =1;
        break
    end
end

if conflict
    choice = questdlg(['This ID=' num2str(newID) ' has already been assigned.'], 'ID conflict',...
    'Assign ID to new segmentation', 'Keep previous ID', 'Assign ID to new segmentation');
    switch choice
        case 'Assign ID to new segmentation'
            p.ID = newID;
            handles.D(d).ID = 0;
        case 'Keep previous ID'
            p.ID = 0;
    end
else
    p.ID = newID;
end


D = p;






%keyboard;




%==========================================================================
% --- Executes on a key press on hFig - the figure containing the image
%==========================================================================
% --- Executes on key press with focus on control_figure and no controls selected.
function keypresshandler(src, evnt, handles)

src = handles.Gtool;
handles = guidata(src);

%disp(['=>you pressed "' evnt.Key '"']);
switch evnt.Key
    case 'rightarrow'
        tvalue = min(get(handles.tslider, 'Max'), get(handles.tslider, 'Value') + 1);
        set(handles.tslider, 'Value', tvalue);
        tslider_Callback(handles.tslider, [ ], handles);
    case 'leftarrow'
        tvalue = max(get(handles.tslider, 'Min'), get(handles.tslider, 'Value') - 1);
        set(handles.tslider, 'Value', tvalue);
        tslider_Callback(handles.tslider, [ ], handles);        
    case 'uparrow'
        zvalue = min(get(handles.zslider, 'Max'), get(handles.zslider, 'Value') + 1);
        set(handles.zslider, 'Value', zvalue);
        zslider_Callback(handles.zslider, [ ], handles);
    case 'downarrow'
        zvalue = max(get(handles.zslider, 'Min'), get(handles.zslider, 'Value') - 1);
        set(handles.zslider, 'Value', zvalue);
        zslider_Callback(handles.zslider, [ ], handles);
    case 'delete'
        pushDelete_Callback(handles.pushDelete, [ ], handles);
    case 'return'
        pushSegment_Callback(handles.pushSegment, [], handles);
    case 'g'
        pushSegGrowth_Callback(handles.pushSegGrowth, [], handles);
    case 'space'
        pushObject_Callback(handles.pushObject, [], handles);
    case 's'
        ann = get(handles.checkAnn, 'Value');
        set(handles.checkAnn, 'Value', ~ann);
        checkAnn_Callback(handles.checkAnn, [], handles);
    case 'h'
        pushHelp_Callback(handles.pushHelp, [], handles);
    case 'a'
        pushAssignID_Callback(handles.pushAssignID, [], handles);
    case 'n'    
        pushNewID_Callback(handles.pushNewID, [], handles);
    case '1'  
        set(handles.listID, 'Value', 1);
        listID_Callback(handles.listID, [], handles);
    case '2'
        if length(get(handles.listID, 'String')) >= 2
            set(handles.listID, 'Value', 2);
            listID_Callback(handles.listID, [], handles);
        end
    case '3'
        if length(get(handles.listID, 'String')) >= 3
            set(handles.listID, 'Value', 3);
            listID_Callback(handles.listID, [], handles);
        end
    case '4'
        if length(get(handles.listID, 'String')) >= 4
            set(handles.listID, 'Value', 4);
            listID_Callback(handles.listID, [], handles);
        end
    case '5'
        if length(get(handles.listID, 'String')) >= 5
            set(handles.listID, 'Value', 5);
            listID_Callback(handles.listID, [], handles);
        end
    case '6'
        if length(get(handles.listID, 'String')) >= 6
            set(handles.listID, 'Value', 6);
            listID_Callback(handles.listID, [], handles);
        end
    case '7'
        if length(get(handles.listID, 'String')) >= 7
            set(handles.listID, 'Value', 7);
            listID_Callback(handles.listID, [], handles);
        end  
    case '8'
        if length(get(handles.listID, 'String')) >= 8
            set(handles.listID, 'Value', 8);
            listID_Callback(handles.listID, [], handles);
        end
    case '9'
        if length(get(handles.listID, 'String')) >= 9
            set(handles.listID, 'Value', 9);
            listID_Callback(handles.listID, [], handles);
        end    
    case 'equal'
        zoomin_Callback(handles.zoomin, [], handles);
    case 'hyphen'
        zoomout_Callback(handles.zoomout, [], handles); 
    otherwise
        disp('unrecognized key command');
end
%guidata(src, handles);




% --- Executes on button press in checkAnn.
function checkAnn_Callback(hObject, eventdata, handles)
% hObject    handle to checkAnn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of checkAnn

handles = updateImage(handles);
guidata(hObject, handles);



function editSigmoid_Callback(hObject, eventdata, handles)
% hObject    handle to editSigmoid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editSigmoid as text
%        str2double(get(hObject,'String')) returns contents of editSigmoid as a double


% --- Executes during object creation, after setting all properties.
function editSigmoid_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editSigmoid (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushDelete.
function pushDelete_Callback(hObject, eventdata, handles)
% hObject    handle to pushDelete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

visrect = handles.api.getVisibleImageRect();
r = round(visrect(2) + visrect(4)/2);
c = round(visrect(1) + visrect(3)/2);

h = imrect(handles.hFigAxis, [c-30 r-30 61 61 ]);
set(h,'Interruptible', 'off');
set(handles.hFig, 'WindowStyle', 'modal');
setColor(h, [.2 .2 .2]);
position = wait(h);

ymin = position(2);
ymax = position(2)+position(4);
xmin = position(1);
xmax = position(1)+position(3);


found = 0;
delete(h); 

dlist = handles.dlist{handles.t};
for d = 1:length(dlist)

    c = handles.D(dlist(d)).Centroid;
    if (c(1) >= xmin) && (c(1) <= xmax) && (c(2) >=ymin) && (c(2) <= ymax)
        found = 1;
        %disp(['kill d=' num2str(dlist(d))]);
        choice = questdlg(['Delete d=' num2str(dlist(d)) '?'], 'Delete an object?',  'Delete', 'Cancel', 'Cancel');
        switch choice
            case 'Delete'
                dlist
                dlist(d)
                handles  = deleteNuclei(handles, dlist(d));
                %disp('delete!!')
                break;
            case 'Cancel'
        end
    end
end

glist = handles.glist{handles.t};
if found == 0
    for g = 1:length(glist)
        c = handles.G(glist(g)).Centroid;
        if (c(1) >= xmin) && (c(1) <= xmax) && (c(2) >=ymin) && (c(2) <= ymax)
            found = 1;
            %disp(['kill d=' num2str(dlist(d))]);
            choice = questdlg(['Delete g=' num2str(glist(g)) '?'], 'Delete an object?',  'Delete', 'Cancel', 'Cancel');
            switch choice
                case 'Delete'
                    glist
                    glist(g)
                    handles  = deleteGrowthCone(handles, glist(g));
                    %disp('delete!!')
                    break;
                case 'Cancel'
            end
        end
    end
end

set(handles.hFig, 'WindowStyle', 'normal');

if ~found
    disp('Nothing to delete!');
end

%keyboard;

handles = updateImage(handles);
handles = updateStats(handles);
guidata(hObject, handles);




function handles  = deleteNuclei(handles, id)

t = handles.D(id).t;
ind = find(handles.dlist{t} == id);
handles.dlist{t}(ind) = [];

% subtract 1 from all later detection entries
for t = 1:handles.tmax
    for i = 1:length(handles.dlist{t})
        if handles.dlist{t}(i) > id
            handles.dlist{t}(i)  =  handles.dlist{t}(i) -1;
        end
    end
end
% delete the detection entry
handles.D(id) = [];


function handles  = deleteGrowthCone(handles, id)

t = handles.G(id).t;
ind = find(handles.glist{t} == id);
handles.glist{t}(ind) = [];

% subtract 1 from all later detection entries
for t = 1:handles.tmax
    for i = 1:length(handles.glist{t})
        if handles.glist{t}(i) > id
            handles.glist{t}(i)  =  handles.glist{t}(i) -1;
        end
    end
end
% delete the detection entry
handles.G(id) = [];


% --- Executes on button press in pushHelp.
function pushHelp_Callback(hObject, eventdata, handles)
% hObject    handle to pushHelp (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

h = helpdlg({'Help for Gtool v0.1', '' , ...
    'DATA. To load a time-series stack, click LOAD and select a folder containing the TIF stacks. Press DENOISE to remove noise from the current stack. Press SAVE to write the stacks to TIFS in a desired folder.', '', ...
    'VISUALIZATION. Press MAX INTENSITY or STD DEV to view the stack projections. Press Z-SLICE to view the stack slice-by-slice. Zoom controls appear on the lower right. Adjust the time step with the scrollbar on the bottom.', '', ...
    'CONTRAST. Original 16-bit data is displayed. Press INVERT to invert the contrast. Press ADJUST to adjust the contrast (do NOT press APPLY TO DATA in dialog). Note: you must press APPLY after pressing ADJUST to keep the contrast settings. RESET returns to the orignal contrast settings.', '', ...
    'NUCLEI. Create a new circular annotation tool by pressing NEW. Reposition and adjust the tool so that it contains the entire nucleus and is well centered. In projection modes, the z-center is selected automatically, in z-slice mode the current slice is the z-center. Press SEGMENT to perform the active contour segmentation. Press DELETE to remove a bad segmentation. A rectangle will appear, all objects within the rectangle will be deleted. Double-click the rectangle to delete.', '', ...
    'SHORTCUTS. When data window is in focus.', '', ...
    'right/left arrow	Time-step forward/back', ...
    'up/down arrow      Z-slice up/down', ...
    'spacebar           New object tool', ...
    'enter              Segment a nucleus', ...
    'g                  Segment a growth cone', ...
    'delete             Delete an object', ...
    's                  Show/Hide annotatinos', ...
    'n                  Add a new neuron ID to the list', ...
    'a                  Assign an ID to existing annotation', ...
    '[1-9]              Shortcut to select neurons 1-9', ...
    },'Help');



function handles = clearAnnotations(handles)

handles.D = [];
handles.D.Area = [];
handles.D.Centroid = [];
handles.D.zCenter = [];
handles.D.Volume = [];
handles.D.meanIntensity = [];
handles.D.stdIntensity = [];
handles.D.MajorAxisLength = [];
handles.D.MinorAxisLength = [];
handles.D.Eccentricity = [];
handles.D.Orientation = [];
handles.D.Perimeter = [];
handles.D.PixelIdxList = [];
handles.D.PerimList =[];
handles.D.slices = {};
handles.D.pslices = {};
handles.D.t =[];
handles.dlist = {};

handles.G = [];
handles.G.Area = [];
handles.G.Centroid = [];
handles.G.zCenter = [];
handles.G.Volume = [];
handles.D.meanIntensity = [];
handles.D.stdIntensity = [];
handles.G.MajorAxisLength = [];
handles.G.MinorAxisLength = [];
handles.G.Eccentricity = [];
handles.G.Orientation = [];
handles.G.Perimeter = [];
handles.G.PixelIdxList = [];
handles.G.PerimList =[];
handles.G.slices = {};
handles.G.pslices = {};
handles.G.t =[];
handles.glist = {};

handles.spacing = [1 1 1];
handles.IDs = 1;


visrect = handles.api.getVisibleImageRect();
r = round(visrect(2) + visrect(4)/2);
c = round(visrect(1) + visrect(3)/2);
diameter = str2double(get(handles.editDiameter, 'String'));
handles.ellipse = imellipse(handles.hFigAxis, [c-(diameter/2) r-(diameter/2) diameter diameter ]);
setFixedAspectRatioMode(handles.ellipse, 1);
delete(handles.ellipse);

if exist('handles.hMeasure', 'var')
    if isvalid(handles.hMeasure)
        delete(handles.hMeasure);
    end
else
    handles.hMeasure = imdistlinescaled(handles.hFigAxis);
    handles.hMeasure.setScale(handles.spacing);
    delete(handles.hMeasure);
end
set(handles.pushMeasure, 'Value', 0);





function handles = saveAnnotations(handles)


if isempty(handles.D(1).Area)
    choice = questdlg('The annotations appear to be empty. Save anyway?', 'Save annotations',...
    'Save', 'Cancel', 'Cancel');
else
    choice = '';
end


switch choice
    case 'Cancel'
        return;
    otherwise


    %folder = uigetfile(handles.folder, 'Select folder to save the annotations');
    [filename, pathname, filterindex] = uiputfile( ...
           {'*.mat','MAT-files (*.mat)'; ...
            '*.mdl','Models (*.mdl)'; ...
            '*.*',  'All Files (*.*)'}, ...
            'Save as', 'Untitled.mat');

    D = handles.D;
    dlist = handles.dlist;
    G = handles.G;
    glist = handles.glist;
    IDs = handles.IDs;

    if pathname ~= 0
        filenm = [pathname filename];
        save(filenm, 'D', 'dlist', 'G', 'glist', 'IDs');
        disp(['saved ' filenm]);
    end
end    
    
    
function [handles pathname] = loadAnnotations(handles)

handles = clearAnnotations(handles);

[filename, pathname, filterindex] = uigetfile( ...
       {'*.mat','MAT-files (*.mat)'; ...
        '*.mdl','Models (*.mdl)'; ...
        '*.*',  'All Files (*.*)'}, ...
        'Pick a file', 'Untitled.mat');

if pathname ~= 0
    A = load([pathname filename]);
    handles.D = A.D;
    handles.dlist = A.dlist;
    handles.G = A.G;
    handles.glist = A.glist;
    handles.IDs = A.IDs;
end



% --- Executes on button press in pushLoadAnn.
function pushLoadAnn_Callback(hObject, eventdata, handles)
% hObject    handle to pushLoadAnn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[handles p] = loadAnnotations(handles);
if p ~= 0
    handles = updateImage(handles);
    handles = updateIDList(handles);
    set(handles.listID, 'Value', 1);
    handles = updateStats(handles);
    guidata(hObject, handles);
end

% --- Executes on button press in pushSaveAnn.
function pushSaveAnn_Callback(hObject, eventdata, handles)
% hObject    handle to pushSaveAnn (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = saveAnnotations(handles);
handles = updateImage(handles);
guidata(hObject, handles);


% --- Executes on button press in pushMeasure.
function pushMeasure_Callback(hObject, eventdata, handles)
% hObject    handle to pushMeasure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isvalid(handles.hMeasure)
    handles.hMeasure = imdistlinescaled(handles.hFigAxis);
    %handles.spacing
    handles.hMeasure.setScale(handles.spacing);
    handles.hMeasure.setLabelTextFormatter('%2.2f microns');
else
    delete(handles.hMeasure);
end
guidata(hObject, handles);


% --- Executes on button press in pushNeuriteStart.
function pushNeuriteStart_Callback(hObject, eventdata, handles)
% hObject    handle to pushNeuriteStart (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on button press in pushNeuriteEnd.
function pushNeuriteEnd_Callback(hObject, eventdata, handles)
% hObject    handle to pushNeuriteEnd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushNeuriteSegment.
function pushNeuriteSegment_Callback(hObject, eventdata, handles)
% hObject    handle to pushNeuriteSegment (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushNewGrowth.
function pushNewGrowth_Callback(hObject, eventdata, handles)
% hObject    handle to pushNewGrowth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%pushObject_Callback(handles.pushObject, [], handles);


if isvalid(handles.ellipse)
    delete(handles.ellipse);
end
    
visrect = handles.api.getVisibleImageRect();
r = round(visrect(2) + visrect(4)/2);
c = round(visrect(1) + visrect(3)/2);

diameter = str2double(get(handles.editDiameter, 'String'));
handles.ellipse = imellipse(handles.hFigAxis, [c-20 r-20 diameter diameter ]);
addNewPositionCallback(handles.ellipse,@(p) title(mat2str(p,3)));
fcn = makeConstrainToRectFcn('imellipse',get(handles.hFigAxis,'XLim'),get(handles.hFigAxis,'YLim'));
setPositionConstraintFcn(handles.ellipse,fcn);   

setFixedAspectRatioMode(handles.ellipse, 1);
currentID = handles.IDs(get(handles.listID, 'Value'));
color = handles.colors(currentID,:);
setColor(handles.ellipse, color);

guidata(hObject, handles);



% --- Executes on button press in pushSegGrowth.
function pushSegGrowth_Callback(hObject, eventdata, handles)
% hObject    handle to pushSegGrowth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isvalid(handles.ellipse)
    disp('Create a new object to segment');
    return;
end

position = handles.ellipse.getPosition();
position = double(position);

rmin = position(2);
cmin = position(1);
rmax = position(2) + position(4);
cmax = position(1) + position(3);

rcen = round(position(2) + position(4)/2);
ccen = round(position(1) + position(3)/2);

rcen2 = round(position(4)/2);
ccen2 = round(position(3)/2);

% get the z-position
if ~get(handles.radSlice, 'Value')
    zdata = [squeeze(handles.Data{handles.t}(rmin:rmax,cmin:cmax,:))];
    zdata = double(zdata);
    zdata = mean(zdata,1);
    zdata = mean(zdata,2);
    zdata = squeeze(zdata);
    [zdata handles] = smoothZselection(zdata, handles);
    [maxi zcen2] = max(zdata);
else
    zcen2 = get(handles.zslider,'Value');
end
    
ImageSpacing = handles.spacing / handles.spacing(1);
disp(['spacing = [' ImageSpacing ']']);


data = double(handles.Data{handles.t}(rmin:rmax, cmin:cmax, :));
seedPoint = [rcen2 ccen2 zcen2];
radius = ceil(position(4)/2);

%% sigmoid param
SigWeight = get(handles.editSigmoidGrowth, 'String');
SigWeight = str2num(SigWeight);

if isvalid(handles.ellipse)
    delete(handles.ellipse);
end

tic
segmentation = SGAC(data, ImageSpacing, seedPoint, radius, SigWeight);
toc

offsets = double([rmin rmax cmin cmax]);
[Gnew handles] = extractGrowthConeSegmentation(data, segmentation, [size(handles.Data{handles.t},1) size(handles.Data{handles.t},2)], offsets, handles);

if ~isempty(Gnew)
    glength = length(handles.G);
    if (glength == 1) && isempty(handles.G.Area)
        Gnew.zCenter = zcen2;
        handles.G = Gnew;
    else
        Gnew.zCenter = zcen2;
        handles.G(glength+1) = Gnew;
    end

    handles.glist{handles.t}(end+1) = length(handles.G);
end

handles = updateImage(handles);
handles = updateStats(handles);
guidata(hObject, handles);




function [G handles] = extractGrowthConeSegmentation(rawdata, V, IMSIZE, offsets, handles)

G = [];

I = zeros(IMSIZE);
Vflat = max(V,[],3);

rmin = offsets(1); rmax = offsets(2); cmin = offsets(3); cmax = offsets(4);
I(rmin:rmax, cmin:cmax) = Vflat;

L = bwlabel(I); 
p = regionprops(L, 'Area', 'Eccentricity', 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter', 'PixelIdxList');

if isempty(p)
    disp('Segmentation was empty!');
    return;
end

arealist = [p.Area];
[area,bigind] = max(arealist);
p = p(bigind);

I = zeros(IMSIZE);
I(p.PixelIdxList) = 1;
perim = bwperim(I);
p.PerimList = find(perim);

% compute interesting statistics
Vbin = V > 0;
p.Volume = sum(Vbin(:));
Ivec = rawdata(Vbin);
p.meanIntensity = mean(Ivec(:));
p.stdIntensity = std(Ivec(:));

p.t = handles.t;
p.ID = 0;

p.slices = cell(1,handles.zmax);
for z = 1:handles.zmax
    I = zeros(IMSIZE);
    I(rmin:rmax, cmin:cmax) = V(:,:,z);
    p.slices{z} = find(I);   
    p.pslices{z} = find(bwperim(I));
end


% check for an ID conflict with the new annotation
newID = handles.IDs(get(handles.listID, 'Value'));
conflict = 0;
glist = handles.glist{handles.t};
for g = glist
    if handles.G(g).ID == newID
        conflict =1;
        break
    end
end

% if conflict
%     choice = questdlg(['This ID=' num2str(newID) ' has already been assigned.'], 'ID conflict',...
%     'Assign ID to new segmentation', 'Keep previous ID', 'Assign ID to new segmentation');
%     switch choice
%         case 'Assign ID to new segmentation'
%             p.ID = newID;
%             handles.G(g).ID = 0;
%         case 'Keep previous ID'
%             p.ID = 0;
%     end
% else
%     p.ID = newID;
% end

p.ID = newID;
G = p;




function editSigmoidGrowth_Callback(hObject, eventdata, handles)
% hObject    handle to editSigmoidGrowth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editSigmoidGrowth as text
%        str2double(get(hObject,'String')) returns contents of editSigmoidGrowth as a double


% --- Executes during object creation, after setting all properties.
function editSigmoidGrowth_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editSigmoidGrowth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listID.
function listID_Callback(hObject, eventdata, handles)
% hObject    handle to listID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listID contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listID

if ~isempty(handles.D(1).Area)
    handles = updateImage(handles);
    handles = updateStats(handles);
    guidata(hObject, handles);
end
    
% --- Executes during object creation, after setting all properties.
function listID_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushNewID.
function pushNewID_Callback(hObject, eventdata, handles)
% hObject    handle to pushNewID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(handles.IDs)
    nextID = 1;
else
    nextID = handles.IDs(end) + 1;
end
handles.IDs = [handles.IDs nextID];

set(handles.listID, 'Value', nextID);

handles = updateIDList(handles);
guidata(hObject, handles);

% --- Executes on button press in pushRemoveID.
function pushRemoveID_Callback(hObject, eventdata, handles)
% hObject    handle to pushRemoveID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if numel(handles.IDs) <= 1
    disp('Cannot remove any more IDs');
    return;
else
    get(handles.listID, 'Value')
    removeID = handles.IDs(get(handles.listID, 'Value'));

    
    handles.IDs(removeID) = [];

    
    if removeID > 1
        set(handles.listID, 'Value', removeID-1);
    end
        
    % clear the IDs of all nuclei with this ID
    for t = 1:handles.tmax
        dlist = handles.dlist{t};
        for d = dlist
            if handles.D(d).ID == removeID
                handles.D(d).ID = 0;
                disp(['set d=' num2str(d) '.ID = 0']);
            end
        end
    end
    % clear the IDs of all growth cones with this ID
    for t = 1:handles.tmax
        glist = handles.glist{t};
        for g = glist
            if handles.G(g).ID == removeID
                handles.G(g).ID = 0;
                disp(['set g=' num2str(g) '.ID = 0']);
            end
        end
    end
    
    % reduce all higher nuclei IDs by 1
    for t = 1:handles.tmax
        dlist = handles.dlist{t};
        for d = dlist
            if handles.D(d).ID > removeID
                handles.D(d).ID = handles.D(d).ID -1;
                disp(['set d=' num2str(d) '.ID from ' num2str(handles.D(d).ID +1) ' to ' num2str(handles.D(d).ID)]);
            end
        end
    end
    % reduce all higher growth cone IDs by 1
    for t = 1:handles.tmax
        glist = handles.glist{t};
        for g = glist
            if handles.G(g).ID > removeID
                handles.G(g).ID = handles.G(g).ID -1;
                disp(['set g=' num2str(g) '.ID from ' num2str(handles.G(g).ID +1) ' to ' num2str(handles.G(g).ID)]);
            end
        end
    end
    
    handles.IDs( handles.IDs > removeID) =    handles.IDs( handles.IDs > removeID) - 1; 
    handles = updateIDList(handles);
end
handles = updateImage(handles);
handles = updateStats(handles);
guidata(hObject, handles);


% --- Executes on button press in pushAssignID.
function pushAssignID_Callback(hObject, eventdata, handles)
% hObject    handle to pushAssignID (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


visrect = handles.api.getVisibleImageRect();
r = round(visrect(2) + visrect(4)/2);
c = round(visrect(1) + visrect(3)/2);

h = imrect(handles.hFigAxis, [c-30 r-30 61 61 ]);
set(h,'Interruptible', 'off');
set(handles.hFig, 'WindowStyle', 'modal');
setColor(h, [.2 .2 .2]);
position = wait(h);

ymin = position(2);
ymax = position(2)+position(4);
xmin = position(1);
xmax = position(1)+position(3);

delete(h); 

% assign found nuclei
found = 0;
dlist = handles.dlist{handles.t};
for d = 1:length(dlist)

    c = handles.D(dlist(d)).Centroid;
    if (c(1) >= xmin) && (c(1) <= xmax) && (c(2) >=ymin) && (c(2) <= ymax)
        found = 1;
        break;
    end
end
d_id = dlist(d);
set(handles.hFig, 'WindowStyle', 'normal');
if ~found
    disp('No nuclei to assign!');
else
    handles = assignNucleiID(handles, d_id);
end

% assign found growth cones
found = 0;
glist = handles.glist{handles.t};
for g = 1:length(glist)

    c = handles.G(glist(g)).Centroid;
    if (c(1) >= xmin) && (c(1) <= xmax) && (c(2) >=ymin) && (c(2) <= ymax)
        found = 1;
        break;
    end
end
g_id = glist(g);
set(handles.hFig, 'WindowStyle', 'normal');
if ~found
    disp('No growth cone to assign!');
else
    handles = assignGrowthConeID(handles, g_id);
end

guidata(hObject, handles);




function handles = updateIDList(handles)

str = {};
for i = 1:numel(handles.IDs)
    num = num2str(handles.IDs(i));
    color = handles.colors(i,:);
    b = round(255 * color);
    b1 = dec2hex(b(1)); if length(b1)==1; b1 = ['0' b1]; end;
    b2 = dec2hex(b(2)); if length(b2)==1; b2 = ['0' b2]; end;
    b3 = dec2hex(b(3)); if length(b3)==1; b3 = ['0' b3]; end;
    colorstring = [b1 b2 b3];
    str{i} = ['<HTML><FONT COLOR=' colorstring '>' num '</FONT></HTML>'];

end
%set(handles.listID, 'String', {'<html><b>1</b></html>', '2','<HTML><FONT COLOR=FF0000>Row 3</FONT><HTML>'});
set(handles.listID, 'String', str);




function handles = assignNucleiID(handles, d)

% make sure it is a valid detection 
try
    isempty(handles.D(d));
catch me
    disp('Error: tried to assign a non-existent detection!');
    return;
end

assignedID = handles.IDs(get(handles.listID, 'Value'));


t = handles.D(d).t;
dlist = handles.dlist{t};

for i = 1:length(dlist)
    d_i = dlist(i);
    
    if handles.D(d_i).ID == assignedID
        handles.D(d_i).ID = 0;
        disp(['reassigned d=' num2str(d_i) ' from ' num2str(assignedID) ' to 0.']);
    end
end

handles.D(d).ID = assignedID;

disp(['assigned d=' num2str(d) ' from 0 to ' num2str(assignedID) ]);
handles = updateImage(handles);
handles = updateStats(handles);


function handles = assignGrowthConeID(handles, g)

% make sure it is a valid detection 
try
    isempty(handles.G(g));
catch me
    disp('Error: tried to assign a non-existent growth cone!');
    return;
end

assignedID = handles.IDs(get(handles.listID, 'Value'));


t = handles.G(g).t;
glist = handles.glist{t};

% for i = 1:length(glist)
%     g_i = glist(i);
%     
%     if handles.G(g_i).ID == assignedID
%         handles.G(g_i).ID = 0;
%         disp(['reassigned g=' num2str(g_i) ' from ' num2str(assignedID) ' to 0.']);
%     end
% end

handles.G(g).ID = assignedID;

disp(['assigned g=' num2str(g) ' from 0 to ' num2str(assignedID) ]);
handles = updateImage(handles);
handles = updateStats(handles);


% --- Executes on button press in pushColor.
function pushColor_Callback(hObject, eventdata, handles)
% hObject    handle to pushColor (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

currentID = handles.IDs(get(handles.listID, 'Value'));
newcolor = uisetcolor;
handles.colors(currentID, :) = newcolor;

handles = updateImage(handles);
handles = updateIDList(handles);
guidata(hObject, handles);




% --- Executes on button press in pushStats.
function pushStats_Callback(hObject, eventdata, handles)
% hObject    handle to pushStats (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if get(hObject, 'Value')
    disp('stats on');
    
    currentID = get(handles.listID, 'Value');
    handles.hStats = figure('Toolbar','none','Menubar','none','NumberTitle', 'off', 'Name', ['Stats for Neuron ID=' num2str(currentID) ]);
    handles = updateStats(handles);
else
    disp('stats off');
    
    h = handles.hStats;
    if ishandle(h)
        close(handles.hStats);
    end
end

guidata(hObject, handles);



function handles = updateStats(handles)

h = handles.hStats;
if ishandle(h)

    currentID = get(handles.listID, 'Value');
    set(handles.hFig, 'Name', ['Stats for Neuron ID=' num2str(currentID) ]);
    if ~isempty(handles.D(1).Area)

        %figure(h);
        set(0,'CurrentFigure',h);
        
        % Nucleus stats
        Nvolume = getStat(handles.D, currentID, 'Volume', 1:handles.tmax) * prod(handles.spacing); 
        subplot(2,3,1); plot(Nvolume); title('Volume (\mu m^3)'); grid on;
        hold on; plot(handles.t, Nvolume(handles.t), 'b.'); hold off;
        
        Nlen = getStat(handles.D, currentID, 'MajorAxisLength', 1:handles.tmax) * handles.spacing(1); 
        subplot(2,3,2); plot(Nlen); title('Length (\mu m)'); grid on;
        hold on; plot(handles.t, Nlen(handles.t), 'b.'); hold off;
        
        NmI = getStat(handles.D, currentID, 'meanIntensity', 1:handles.tmax);
        subplot(2,3,3); plot(NmI); title('Mean Intensity'); grid on;
        hold on; plot(handles.t, NmI(handles.t), 'b.'); hold off;
        
        NsI = getStat(handles.D, currentID, 'stdIntensity', 1:handles.tmax);
        subplot(2,3,6); plot(NsI); title('S. Dev. Intensity'); grid on;
        hold on; plot(handles.t, NsI(handles.t), 'b.'); hold off;
        
        Necc = getStat(handles.D, currentID, 'Eccentricity', 1:handles.tmax);
        subplot(2,3,4); plot(Necc); title('Elongation'); grid on;
        hold on; plot(handles.t, Necc(handles.t), 'b.'); hold off;
        
        Nvel = getDistanceTraveled(handles.D, currentID, 1:handles.tmax) * handles.spacing(1); 
        subplot(2,3,5); plot(Nvel); title('Velocity (\mu m / frame)'); grid on;
        hold on; plot(handles.t, Nvel(handles.t), 'b.'); hold off;
        
        figure(handles.hFig);
        
    else
        set(0,'CurrentFigure',h);
        subplot(2,3,1); cla;
        subplot(2,3,2); cla;
        subplot(2,3,3); cla;
        subplot(2,3,4); cla;
        subplot(2,3,5); cla;
        subplot(2,3,6); cla;
        figure(handles.hFig);
    end
   
end



function editDiameter_Callback(hObject, eventdata, handles)
% hObject    handle to editDiameter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of editDiameter as text
%        str2double(get(hObject,'String')) returns contents of editDiameter as a double


% --- Executes during object creation, after setting all properties.
function editDiameter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to editDiameter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
