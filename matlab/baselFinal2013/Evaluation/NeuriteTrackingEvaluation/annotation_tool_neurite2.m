function varargout = annotation_tool_neurite2(varargin)
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

% Last Modified by GUIDE v2.5 09-Aug-2013 01:20:35

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

handles.hFig        = figure('Toolbar','none','Menubar','none', 'Position', [2323 4 1427 1025]);
% handles.hFig        = figure('Position', [1937 262 696 520]);
handles.hIm         = imshow(zeros(1500)); %imshow(imread('peppers.png'));
handles.hSP         = imscrollpanel(handles.hFig,handles.hIm);
handles.api         = iptgetapi(handles.hSP);
handles.hFigAxis    = gca;
handles.mv          = {};
handles.seq         = [];
handles.zoom        = 2;
handles.TR          = [];
handles.previous_t  = 1;
handles.cellpatch   = [];
handles.celltext    = [];



set(handles.figure1, 'Position', [333 29.2857 50.6667 43.5714]);
handles.api.setMagnification(handles.zoom);
set(handles.hSP,'Units','normalized','Position',[0 0 1 1])
set(handles.hFig,'KeyPressFcn',{@keypresshandler, handles}); % set an event listener for keypresses
handles.ovpanel = imoverviewpanel(handles.ovpanel, handles.hIm);



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
mv = loadMovie(Gfolder);
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


handles.colors = color_list();


% load the Neurite Ground Truth
folder = '/home/ksmith/data/sinergia_evaluation/annotations/neuritetool/';
filename = sprintf('%s%s.mat', folder, handles.seq);
if exist(filename, 'file')
    fprintf('loading from %s\n', filename);
    load(filename);
    handles.NGT = NGT;
    handles.CELL = NEST;
else    
    % load cell tracking data
    cell_folder = '/home/ksmith/data/sinergia_evaluation/annotations/celltracking/';
    cell_filename = sprintf('%sannotation%s.mat', cell_folder, handles.seq);
    load(cell_filename);
    handles.CELL = EST;
    handles = generateCELLannotations(handles);
    handles.NGT = handles.CELL;    
    handles = cleanGT(handles);
%     % auto-generate Ground truth from CELL
%     
end
    




% show the current image
guidata(hObject, handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
slider_TR_Callback(handles.slider_TR, [ ], handles);
guidata(hObject, handles);









function handles = show_current_image(handles)

t = round(get(handles.tslider,'Value'));
I = handles.mv{t};

% figure(handles.hFig);
% axis(handles.hFigAxis);


% remove old cell drawings and text.
% a = findall(handles.hFigAxis);
% for i = 2:numel(a)
%     if isgraphics(a(i))
%         delete(a(i));
%     end
% end

% colorGT = [1 .3 1];
colorGT = [1 .3 .3];
colorOKest = [.3 .3 1];
colorERRORest = [1 .3 .3];

a = findobj('Color', colorOKest);
for i = 1:numel(a)
    delete(a(i));
end
a = findobj('FaceColor', colorOKest);
for i = 1:numel(a)
    delete(a(i));
end
a = findobj('EdgeColor', colorOKest);
for i = 1:numel(a)
    delete(a(i));
end

a = findobj('FaceColor', colorERRORest);
for i = 1:numel(a)
    delete(a(i));
end
a = findobj('Color', colorERRORest);
for i = 1:numel(a)
    delete(a(i));
end

a = findobj('FaceColor', colorGT);
for i = 1:numel(a)
    delete(a(i));
end
a = findobj('EdgeColor', colorGT);
for i = 1:numel(a)
    delete(a(i));
end


%% render the tracking results
TRid = get(handles.slider_TR, 'Value');
% fprintf('rendering t=%d  TRid=%d \n', t, TRid);



if TRid > 0
    switch handles.CELL(TRid).P(t).status
        case 'OK'
            
            handles.cellpatch = patch(handles.CELL(TRid).P(t).x, handles.CELL(TRid).P(t).y, 1, 'FaceColor', colorOKest, 'FaceAlpha', .5, 'EdgeColor', [1 1 1], 'EdgeAlpha', .5);
            str = sprintf('%d',TRid);
            handles.celltext = text(max(handles.CELL(TRid).P(t).x)+5, min(handles.CELL(TRid).P(t).y), str, 'Color', colorOKest);
        case {'FP', 'FN', 'MT', 'MO', 'FI'}
            
            handles.cellpatch = patch(handles.CELL(TRid).P(t).x, handles.CELL(TRid).P(t).y, 1, 'FaceColor', colorERRORest, 'FaceAlpha', .5, 'EdgeColor', [0 0 0], 'EdgeAlpha', .5);
            str = sprintf('%d',TRid);
            handles.celltext = text(max(handles.CELL(TRid).P(t).x)+5, min(handles.CELL(TRid).P(t).y), str, 'Color', colorERRORest);
        otherwise
            
    end
    
    hide_flag = get(handles.hide_button, 'Value');
    if (~hide_flag) && isfield(handles.CELL(TRid).P(t), 'N')
        for n = 1:numel(handles.CELL(TRid).P(t).N)
            x = handles.CELL(TRid).P(t).N(n).x;
            y = handles.CELL(TRid).P(t).N(n).y;
            
            h = patchline(x,y,ones(size(x)),'linestyle','-','edgecolor',colorOKest,'linewidth',3,'edgealpha',0.6);
        end
    end
        

     hide_flag = get(handles.toggle_hide_GT, 'Value'); 
     if (~hide_flag) && isfield(handles.NGT(TRid).P(t), 'N')
        for n = 1:numel(handles.NGT(TRid).P(t).N)
            x = handles.NGT(TRid).P(t).N(n).x;
            y = handles.NGT(TRid).P(t).N(n).y;
            GTWIDTH = 5;
            
            h = patchline(x,y,ones(size(x)),'linestyle','-','edgecolor',colorGT,'linewidth',GTWIDTH,'edgealpha',0.5);
        end
        
    end
end


% figure(handles.hFig);
% axis(handles.hFigAxis);

fprintf('rendering t=%d  TRid=%d\n', t, TRid);

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
t = round(get(handles.tslider,'Value'));
fprintf('dev test t = %d\n', t);


keyboard;





%==========================================================================
% --- Executes on a key press on hFig - the figure containing the image
%==========================================================================
% --- Executes on key press with focus on control_figure and no controls selected.
function keypresshandler(src, evnt, handles)

mother_handle = handles.figure1;
handles = guidata(mother_handle);

disp(['=>you pressed "' evnt.Key '"']);
switch evnt.Key
    case 'rightarrow'
        tvalue = min(get(handles.tslider, 'Max'), get(handles.tslider, 'Value') + 1);
        set(handles.tslider, 'Value', tvalue);
        str = sprintf('%d/%d', tvalue, get(handles.tslider, 'Max'));
        set(handles.ttxt, 'String', str);
    case 'leftarrow'
        tvalue = max(get(handles.tslider, 'Min'), get(handles.tslider, 'Value') - 1);
        set(handles.tslider, 'Value', tvalue);
        str = sprintf('%d/%d', tvalue, get(handles.tslider, 'Max'));
        set(handles.ttxt, 'String', str);
    case 'pagedown'
        hide_value = get(handles.toggle_hide_GT, 'Value');
        set(handles.toggle_hide_GT, 'Value', ~hide_value);
%         hide_button_Callback(handles.hide_button, [ ], handles);
    case 'pageup'
        hide_value = get(handles.hide_button, 'Value');
        set(handles.hide_button, 'Value', ~hide_value);
%         hide_button_Callback(handles.hide_button, [ ], handles);
    case 'control'
        handles = add_long_neurite(handles);
    case 'downarrow'  %'pageup'
        handles = remove_a_neurite(handles);
    case 'uparrow'  %'pagedown'
        handles = add_a_neurite(handles);
    case 'shift'
        handles = copy_CELL_to_NGT(handles);
    case 'space'
        handles = entireCELLtoNGT(handles);
end

handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(mother_handle, handles);


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


function handles = updateAxes2(handles)

t = round(get(handles.tslider,'Value'));
Tmax = get(handles.tslider, 'Max');
TRid = get(handles.slider_TR, 'Value');
% TRTimeList = [handles.TR.TrackedCells(TRid).TimeStep(:).Time];


TRTimeList = [];
BADList = [];
for i = 1:numel(handles.CELL(TRid).P)
    switch handles.CELL(TRid).P(i).status
        case 'OK'
            TRTimeList = [TRTimeList; i];
        case {'FP', 'FN', 'MT', 'MO', 'FI'}
            BADList = [BADList; i];
        otherwise
            % do nothing
    end
end

cla(handles.traxis);
plot(handles.traxis,TRTimeList, zeros(size(TRTimeList)), 'b.');
hold(handles.traxis, 'on');
plot(handles.traxis,BADList, zeros(size(BADList)), 'r.');
plot(handles.traxis,[t t], [-.5 .5], 'k-');
hold(handles.traxis, 'off');

axis(handles.traxis,[0 Tmax+1 -.5 .5]);




% --- Executes on button press in button_save.
function button_save_Callback(hObject, eventdata, handles)
% hObject    handle to button_save (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% fprintf('need to write save function\n');
folder = '/home/ksmith/data/sinergia_evaluation/annotations/neuritetool/';
filename = sprintf('%s%s.mat', folder, handles.seq);
fprintf('saving to %s\n', filename);

NGT = handles.NGT;
NEST = handles.CELL;
save(filename, 'NGT', 'NEST');



% --- Executes on button press in hide_button.
function hide_button_Callback(hObject, eventdata, handles)
% hObject    handle to hide_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of hide_button


fprintf('hide is %d\n', get(hObject, 'Value'));

%handles.hide_neurite= get(handles.hide_button, 'Value');
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(hObject, handles);



% --- Executes on button press in button_generateGT.
function button_generateGT_Callback(hObject, eventdata, handles)
% hObject    handle to button_generateGT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = entireCELLtoNGT(handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(hObject, handles);


function handles = entireCELLtoNGT(handles)
Tmax = round(get(handles.tslider,'Max'));
i = get(handles.slider_TR, 'Value');

for t = 1:Tmax
    num_neurites = numel(handles.CELL(i).P(t).N);
    fprintf('copying CELL(%d) -> NGT(%d)  t=%d\n', i, i,t);
    for n = 1:num_neurites
        handles.NGT(i).P(t).N(n).x = handles.CELL(i).P(t).N(n).x;
        handles.NGT(i).P(t).N(n).y = handles.CELL(i).P(t).N(n).y;
    end
end


function handles = generateCELLannotations(handles)
for i = 1:numel(handles.CELL)
    fprintf('adding neurites to CELL %d\n', i);
    cellTimeList = [handles.TR.TrackedCells(i).TimeStep(:).Time];
    for t = 1:numel(handles.CELL(i).P)
        t_index = find(cellTimeList == t);
        switch handles.CELL(i).P(t).status
            case 'OK'
                num_neurites = numel(handles.TR.TrackedCells(i).TimeStep(t_index).NeuritesList);
                for n = 1:num_neurites
                    Parents = handles.TR.TrackedCells(i).TimeStep(t_index).NeuritesList(n).Parents;
                    NeuritePixelIdxList = handles.TR.TrackedCells(i).TimeStep(t_index).NeuritesList(n).NeuritePixelIdxList;
                    NumKids = handles.TR.TrackedCells(i).TimeStep(t_index).NeuritesList(n).NumKids;

                    [neurite_r, neurite_c] = drawNeurite(Parents,NeuritePixelIdxList,NumKids);

                    handles.CELL(i).P(t).N(n).x = neurite_c;
                    handles.CELL(i).P(t).N(n).y = neurite_r;
                end
            otherwise
        end
    end
end

function handles = cleanGT(handles)
for i = 1:numel(handles.NGT)
    fprintf('adding neurites to CELL %d\n', i);
    for t = 1:numel(handles.NGT(i).P)
        handles.NGT(i).P(t).N(1).x = 1;
        handles.NGT(i).P(t).N(1).y = 1;
        handles.NGT(i).P(t).N(:) = [];
    end
end


function handles = copy_CELL_to_NGT(handles)

t = round(get(handles.tslider,'Value'));
i = get(handles.slider_TR, 'Value');
num_neurites = numel(handles.CELL(i).P(t).N);
for n = 1:num_neurites
    handles.NGT(i).P(t).N(n).x = handles.CELL(i).P(t).N(n).x;
    handles.NGT(i).P(t).N(n).y = handles.CELL(i).P(t).N(n).y;
end


% --- Executes on button press in button_add.
function button_add_Callback(hObject, eventdata, handles)
% hObject    handle to button_add (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

fprintf('add a Neurite GT\n');


% --- Executes on button press in button_subtract.
function button_subtract_Callback(hObject, eventdata, handles)
% hObject    handle to button_subtract (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = remove_a_neurite(handles);
handles = show_current_image(handles);
handles = updateAxes2(handles);
guidata(hObject, handles);


function handles = remove_a_neurite(handles)

[x_click,y_click] = ginput(1);
t = round(get(handles.tslider,'Value'));
i = get(handles.slider_TR, 'Value');
num_neurites = numel(handles.NGT(i).P(t).N);
if num_neurites >= 1
    dist_to_click = zeros(num_neurites,1);
    for n = 1:num_neurites  
        x = handles.NGT(i).P(t).N(n).x;
        y = handles.NGT(i).P(t).N(n).y;
        dist_vec = sqrt( (x -x_click).^2 + (y - y_click).^2);
        [min_dist, min_ind] = min(dist_vec);     %#ok<NASGU>
        dist_to_click(n) = min_dist;
    end
    [val, closest_ind] = min(dist_to_click); %#ok<ASGLU>
    fprintf('remove neurite GT %d near x = %1.2f   y = %1.2f\n', closest_ind, x_click,y_click);
    
    % remove the selected neurite
    handles.NGT(i).P(t).N(closest_ind) = [];
else
    fprintf('no neurite to remove\n');
end


function handles = add_a_neurite(handles)
t = round(get(handles.tslider,'Value'));
i = get(handles.slider_TR, 'Value');
num_neurites = numel(handles.NGT(i).P(t).N);
[x_click,y_click] = ginput(3);
set(gcf, 'WindowButtonMotionFcn', @mouseMove);
set(gcf, 'WindowButtonMotionFcn', []);
new_ind = num_neurites + 1;
handles.NGT(i).P(t).N(new_ind).x = x_click;
handles.NGT(i).P(t).N(new_ind).y = y_click;



function handles = add_long_neurite(handles)
t = round(get(handles.tslider,'Value'));
i = get(handles.slider_TR, 'Value');
num_neurites = numel(handles.NGT(i).P(t).N);
[x_click,y_click] = ginput();
set(gcf, 'WindowButtonMotionFcn', @mouseMove);
set(gcf, 'WindowButtonMotionFcn', []);
new_ind = num_neurites + 1;
handles.NGT(i).P(t).N(new_ind).x = x_click;
handles.NGT(i).P(t).N(new_ind).y = y_click;




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
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on button press in toggle_hide_GT.
function toggle_hide_GT_Callback(hObject, eventdata, handles)
% hObject    handle to toggle_hide_GT (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of toggle_hide_GT
