function varargout = annotation_tool(varargin)
% ANNOTATION_TOOL MATLAB code for annotation_tool.fig
%      ANNOTATION_TOOL, by itself, creates a new ANNOTATION_TOOL or raises the existing
%      singleton*.
%
%      H = ANNOTATION_TOOL returns the handle to a new ANNOTATION_TOOL or the handle to
%      the existing singleton*.
%
%      ANNOTATION_TOOL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ANNOTATION_TOOL.M with the given input arguments.
%
%      ANNOTATION_TOOL('Property','Value',...) creates a new ANNOTATION_TOOL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before annotation_tool_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to annotation_tool_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help annotation_tool

% Last Modified by GUIDE v2.5 05-Aug-2013 00:52:17

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @annotation_tool_OpeningFcn, ...
                   'gui_OutputFcn',  @annotation_tool_OutputFcn, ...
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


% --- Executes just before annotation_tool is made visible.
function annotation_tool_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to annotation_tool (see VARARGIN)

% Choose default command line output for annotation_tool
handles.output = hObject;

handles.hFig        = figure('Toolbar','none','Menubar','none');
handles.hIm         = imshow(zeros(1500)); %imshow(imread('peppers.png'));
handles.hSP         = imscrollpanel(handles.hFig,handles.hIm);
handles.api         = iptgetapi(handles.hSP);
handles.hFigAxis    = gca;
handles.hrect       = imrect(handles.hFigAxis, [10 10 100 100]);
handles.mv          = {};
handles.seq         = [];
handles.zoom        = 2;
handles.TR          = [];
handles.GT          = {};
handles.previous_t  = 1;

set(handles.hrect, 'Visible', 'off');
set(handles.hFig, 'Position', [2323 4 1427 1025]);
set(handles.figure1, 'Position', [333 29.2857 50.6667 43.5714]);
handles.api.setMagnification(handles.zoom);
set(handles.hSP,'Units','normalized','Position',[0 0 1 1])
set(handles.hFig,'KeyPressFcn',{@keypresshandler, handles}); % set an event listener for keypresses
handles.ovpanel = imoverviewpanel(handles.ovpanel, handles.hIm);

addNewPositionCallback(handles.hrect, @(p) rectMoveHandler(p,handles));

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes annotation_tool wait for user response (see UIRESUME)
% uiwait(handles.figure1);



% --- Executes on button press in load_button.
function load_button_Callback(hObject, eventdata, handles)
% hObject    handle to load_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% read the images and adjust them
folder = uigetdir('/home/ksmith/data/sinergia_evaluation/', 'Select a dataset');
Gfolder = [folder '/'];
Gfiles = dir([Gfolder '*.TIF']);
IntensityAjustmentGreen.MED = 2537;
IntensityAjustmentGreen.STD = 28.9134;
IntensityAjustmentGreen.MAX = 11234;
if ~exist('TMAX', 'var'); TMAX =  length(Gfiles); end; % number of time steps
if TMAX~=length(Gfiles)
   disp(['problem with data in directory: ' folder]);
   return;
end
[Green, Green_Original] = trkReadImagesAndNormalize(TMAX, Gfolder, IntensityAjustmentGreen);
mv = cell(size(Green));
B = zeros(size(Green{1},1), size(Green{1},2));
TMAX = length(Green);
parfor t = 1:TMAX
    I = double(Green{t});
    I = 1- mat2gray(I);
    Ir = I; Ig = I; Ib = I;
    I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
    mv{t} = I;
end
handles.mv = mv;
handles.I = mv{1};

% set up the t-slider
set(handles.tslider, 'Value', 1);
Tmin = 1;
Tmax = numel(mv);
sliderstep(1) = 1/(Tmax - Tmin); sliderstep(2) = sliderstep(1);
set(handles.tslider,'sliderstep',sliderstep, 'max', Tmax,'min', Tmin,'Value',Tmin);


% load the tracking data
handles.seq = folder(end-8:end-6);
tracker_folder = '/home/ksmith/data/sinergia_evaluation/Detections10x/';
tracker_filename = sprintf('%s%s.mat', tracker_folder, handles.seq);
load(tracker_filename);
handles.TR = Sequence;


% set up the slider_TR
set(handles.slider_TR, 'Value', 1);
TRmin = 1;
TRmax = numel(handles.TR.TrackedCells);
sliderstep(1) = 1/(TRmax - TRmin); sliderstep(2) = sliderstep(1);
set(handles.slider_TR,'sliderstep',sliderstep, 'max', TRmax,'min', TRmin,'Value',TRmin);


% load the GT data / or generate from TR
gt_folder = '/home/ksmith/data/sinergia_evaluation/annotations/';
handles.gt_filename = sprintf('%s%s.mat', gt_folder,handles.seq);
if exist(handles.gt_filename, 'file');
    fprintf('loading GT from %s\n', handles.gt_filename);
    load(handles.gt_filename);
    handles.GT = GT;
else
    for i = 1:TRmax
        handles = generate_GT_from_TR(i,i,handles);
    end
end
max_GTid = numel(handles.GT);
min_GTid = 1;
GTid = 1;
set(handles.slider_GT, 'Value', GTid);
sliderstep(1) = 1/(max_GTid - min_GTid); sliderstep(2) = sliderstep(1);
set(handles.slider_GT,'sliderstep',sliderstep, 'max', max_GTid,'min',min_GTid);


% show the current image
guidata(hObject, handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
slider_TR_Callback(handles.slider_TR, [ ], handles);
slider_GT_Callback(handles.slider_GT, [ ], handles);
guidata(hObject, handles);









function handles = show_current_image(handles)

t = round(get(handles.tslider,'Value'));
I = handles.mv{t};

%% render the tracking results
TRid = get(handles.slider_TR, 'Value');
if TRid > 0
    cellTimeList = [handles.TR.TrackedCells(TRid).TimeStep(:).Time];
    t_index = find(cellTimeList == t);
    if ~isempty(t_index)
        B = zeros(size(I,1),size(I,2));
        Ir = I(:,:,1); Ig = I(:,:,2); Ib = I(:,:,3);
        color = [1 0 0];
        
        SomaM = B > Inf;
        SomaM(handles.TR.TrackedCells(TRid).TimeStep(t_index).SomaPixelIdxList) = 1;
        
        SomaP = bwmorph(SomaM, 'remove');
        SomaP = bwmorph(SomaP, 'dilate');
        SomaP = bwmorph(SomaP, 'thin',1);
        Ir(SomaP) = max(0, color(1) - .2);
        Ig(SomaP) = max(0, color(2) - .2);
        Ib(SomaP) = max(0, color(3) - .2);
        
        % % color the nucleus
        % Ir(handles.TR.TrackedCells(TRid).TimeStep(t).NucleusPixelIdxList) = color(1);
        % Ig(handles.TR.TrackedCells(TRid).TimeStep(t).NucleusPixelIdxList) = color(2);
        % Ib(handles.TR.TrackedCells(TRid).TimeStep(t).NucleusPixelIdxList) = color(3);
        
        I(:,:,1) = Ir; I(:,:,2) = Ig; I(:,:,3) = Ib;
        
    else
%         fprintf('rendering t=%d  t_prev=%d  TRid=does not exist\n', t,handles.previous_t);
    end
end

%% render the current annotation
GTid = get(handles.slider_GT, 'Value');
GTrect = handles.GT{GTid}(t,:);
if GTrect(1) ~= 0
   set(handles.hrect, 'Visible', 'on'); 
%    set(handles.hrect, 'Position', GTrect); 
   handles.hrect.setPosition(GTrect);
else
    set(handles.hrect, 'Visible', 'off');
end




fprintf('rendering t=%d  t_prev=%d  TRid=%d  GTid=%d\n', t, handles.previous_t, TRid, GTid);

handles.I = I;
handles.api.replaceImage(handles.I, 'PreserveView', 1);
% guidata(hObject, handles);



% --- Executes on button press in zoom_minus.
function zoom_minus_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_minus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.zoom = handles.zoom/2;
handles.api.setMagnification(handles.zoom);
guidata(hObject, handles);

% --- Executes on button press in zoom_1.
function zoom_1_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.zoom = 2;
handles.api.setMagnification(handles.zoom);
guidata(hObject, handles);

% --- Executes on button press in zoom_plus.
function zoom_plus_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_plus (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.zoom = handles.zoom*2;
handles.api.setMagnification(handles.zoom);
guidata(hObject, handles);

% --- Executes on button press in dev_button.
function dev_button_Callback(hObject, eventdata, handles)
% hObject    handle to dev_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles

keyboard;





%==========================================================================
% --- Executes on a key press on hFig - the figure containing the image
%==========================================================================
% --- Executes on key press with focus on control_figure and no controls selected.
function keypresshandler(src, evnt, handles)

src = handles.figure1;
handles = guidata(src);

disp(['=>you pressed "' evnt.Key '"']);
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
        zoom_plus_Callback(handles.zoom_plus, [ ], handles);
    case 'downarrow'
        zoom_minus_Callback(handles.zoom_minus, [ ], handles);
    case 'hyphen'
        button_GTsubtract_Callback(handles.button_GTsubtract, [ ], handles);
    case 'equal'
        button_GTadd_Callback(handles.button_GTadd, [ ], handles);    
end



% --- Executes on slider movement.
function slider_TR_Callback(hObject, eventdata, handles)
% hObject    handle to slider_TR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

TRid = round(get(hObject,'Value'));
set(hObject,'Value',TRid);
str = sprintf('TR %d/%d', TRid, get(hObject, 'Max'));
set(handles.txt_TR, 'String', str);
guidata(hObject, handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(hObject, handles);


% --- Executes on slider movement.
function tslider_Callback(hObject, eventdata, handles)
% hObject    handle to tslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

t = round(get(hObject,'Value'));
set(hObject, 'Value', t);
guidata(hObject, handles);
str = sprintf('%d/%d', t, get(hObject, 'Max'));
set(handles.ttxt, 'String', str);
handles.previous_t = t;
guidata(hObject, handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(hObject, handles);


% --- Executes on slider movement.
function slider_GT_Callback(hObject, eventdata, handles)
% hObject    handle to slider_GT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

GTid = round(get(hObject,'Value'));
set(hObject,'Value',GTid);
max_GTid = get(hObject, 'Max');
str = sprintf('GT %d/%d', GTid, max_GTid);
set(handles.txt_GT, 'String', str);
guidata(hObject, handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(hObject, handles);

% --- Executes on button press in button_NewGT.
function button_NewGT_Callback(hObject, eventdata, handles)
% hObject    handle to button_NewGT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

t = round(get(handles.tslider,'Value'));
tmax = get(handles.tslider, 'Max');

max_GTid = get(handles.slider_GT, 'Max');
min_GTid = get(handles.slider_GT, 'Min');

GTid = max_GTid + 1;
max_GTid = GTid;
handles.GT{GTid} = zeros(tmax,4);


set(handles.slider_GT, 'Value', GTid);
sliderstep(1) = 1/(max_GTid - min_GTid); sliderstep(2) = sliderstep(1);
set(handles.slider_GT,'sliderstep',sliderstep, 'max', GTid,'min',min_GTid);

str = sprintf('GT %d/%d', GTid, max_GTid);
set(handles.txt_GT, 'String', str);

guidata(hObject, handles);


% --- Executes on button press in button_GT_from_TR.
function button_GT_from_TR_Callback(hObject, eventdata, handles)
% hObject    handle to button_GT_from_TR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


TRid = get(handles.slider_TR, 'Value');
GTid = get(handles.slider_GT, 'Value');
handles = generate_GT_from_TR(TRid,GTid,handles);

guidata(handles.figure1,handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(handles.figure1,handles);


function handles = generate_GT_from_TR(TRid,GTid,handles)

fprintf('converting TRid=%d data -> GTid=%d data\n', TRid, GTid);
Tmax = get(handles.tslider, 'Max');

cellTimeList = [handles.TR.TrackedCells(TRid).TimeStep(:).Time];
for t = 1:Tmax     
    R = size(handles.I,1); C = size(handles.I,2);
%     B = zeros(R,C);
    t_index = find(cellTimeList == t);
    if ~isempty(t_index)
        indlist = handles.TR.TrackedCells(TRid).TimeStep(t_index).SomaPixelIdxList;
%         rlist = zeros(size(indlist));
%         clist = zeros(size(indlist));
        [rlist clist] = ind2sub([R C], indlist);
        rmin = min(rlist); 
        rmax = max(rlist);
        cmin = min(clist);
        cmax = max(clist);
        
        width = cmax - cmin;
        height = rmax - rmin;
        rect = [cmin rmin width height];
        handles.GT{GTid}(t,:) = rect;
    else
       handles.GT{GTid}(t,:) = [0 0 0 0]; 
    end
end

handles = smoothGT(GTid,handles);
guidata(handles.figure1,handles);

% guidata(hObject, handles);
% guidata(hObject, handles);




function handles = smoothGT(GTid,handles)

fprintf('Smoothing GTid = %d\n', GTid);
Tmax = get(handles.tslider, 'Max');
% smooth the data
inds = find(handles.GT{GTid}(:,1) ~= 0);
tlist = inds;
cvals = handles.GT{GTid}(inds,1);
rvals = handles.GT{GTid}(inds,2);
wvals = handles.GT{GTid}(inds,3);
hvals = handles.GT{GTid}(inds,4);
sf1 = fit(tlist,cvals, 'smoothingspline');
sf2 = fit(tlist,rvals, 'smoothingspline');
sf3 = fit(tlist,wvals, 'smoothingspline');
sf4 = fit(tlist,hvals, 'smoothingspline');

% if GTid == 10
%     keyboard;
% end

t1 = tlist(1);
t2 = tlist(end);
windowT = t1:t2;
cwin = feval(sf1, windowT); clist=zeros(Tmax,1); clist(t1:t2) = cwin;
rwin = feval(sf2, windowT); rlist=zeros(Tmax,1); rlist(t1:t2) = rwin;
wwin = feval(sf3, windowT); wlist=zeros(Tmax,1); wlist(t1:t2) = wwin;
hwin = feval(sf4, windowT); hlist=zeros(Tmax,1); hlist(t1:t2) = hwin;

handles.GT{GTid} = [clist(:) rlist(:) wlist(:) hlist(:)];


function handles = updateAxes2(handles)

t = round(get(handles.tslider,'Value'));
Tmax = get(handles.tslider, 'Max');
TRid = get(handles.slider_TR, 'Value');
TRTimeList = [handles.TR.TrackedCells(TRid).TimeStep(:).Time];
GTid = get(handles.slider_GT, 'Value');
GTTimeList = find(handles.GT{GTid}(:,1) ~= 0);
% axis(handles.traxis); cla;
cla(handles.traxis);
plot(handles.traxis,TRTimeList, zeros(size(TRTimeList)), 'r.');
hold(handles.traxis, 'on');
plot(handles.traxis,[t t], [-.5 .5], 'k-');
hold(handles.traxis, 'off');
% axis(handles.gtaxis); cla;
cla(handles.gtaxis);
plot(handles.gtaxis,GTTimeList, zeros(size(GTTimeList)), 'b.');
hold(handles.gtaxis, 'on');
plot(handles.gtaxis,[t t], [-.5 .5], 'k-');
hold(handles.gtaxis, 'off');
axis(handles.traxis,[0 Tmax+1 -.5 .5]);
axis(handles.gtaxis,[0 Tmax+1 -.5 .5]);
% axis(handles.hFigAxis);



function rectMoveHandler(rect,handles)


src = handles.figure1;
handles = guidata(src);
t = round(get(handles.tslider,'Value'));
GTid = get(handles.slider_GT, 'Value');
% fprintf('the rect moved to %1.2f %1.2f %1.2f %1.2f\n', rect(1), rect(2),rect(3),rect(4));
handles.GT{GTid}(t,:) = rect;
guidata(handles.figure1,handles);




% --- Executes on button press in button_smooth.
function button_smooth_Callback(hObject, eventdata, handles)
% hObject    handle to button_smooth (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

GTid = get(handles.slider_GT, 'Value');
handles = smoothGT(GTid,handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(handles.figure1,handles);

% --- Executes on button press in button_GTadd.
function button_GTadd_Callback(hObject, eventdata, handles)
% hObject    handle to button_GTadd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

t = round(get(handles.tslider,'Value'));
GTid = get(handles.slider_GT, 'Value');
R = size(handles.I,1); C = size(handles.I,2);
rect = [C/2 R/2 50 50];
handles.GT{GTid}(t,:) = rect;
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(handles.figure1,handles);
fprintf('added %1.2f %1.2f %1.2f %1.2f\n', rect(1), rect(2),rect(3),rect(4));

% --- Executes on button press in button_GTsubtract.
function button_GTsubtract_Callback(hObject, eventdata, handles)
% hObject    handle to button_GTsubtract (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

t = round(get(handles.tslider,'Value'));
GTid = get(handles.slider_GT, 'Value');
rect = [0 0 0 0];
handles.GT{GTid}(t,:) = [0 0 0 0];
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(handles.figure1,handles);
fprintf('removed %1.2f %1.2f %1.2f %1.2f\n', rect(1), rect(2),rect(3),rect(4));


% --- Executes on button press in button_save.
function button_save_Callback(hObject, eventdata, handles)
% hObject    handle to button_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

GT = handles.GT;
fprintf('saving GT to %s\n', handles.gt_filename);
save(handles.gt_filename, 'GT');







% --- Executes during object creation, after setting all properties.
function tslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to tslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Outputs from this function are returned to the command line.
function varargout = annotation_tool_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes during object creation, after setting all properties.
function slider_TR_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_TR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

% --- Executes during object creation, after setting all properties.
function slider_GT_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider_GT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



