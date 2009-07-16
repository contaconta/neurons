function varargout = view_spedges(varargin)
% VIEW_SPEDGES M-file for view_spedges.fig
%      VIEW_SPEDGES, by itself, creates a new VIEW_SPEDGES or raises the existing
%      singleton*.
%
%      H = VIEW_SPEDGES returns the handle to a new VIEW_SPEDGES or the handle to
%      the existing singleton*.
%
%      VIEW_SPEDGES('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VIEW_SPEDGES.M with the given input arguments.
%
%      VIEW_SPEDGES('Property','Value',...) creates a new VIEW_SPEDGES or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before view_spedges_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to view_spedges_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help view_spedges

% Last Modified by GUIDE v2.5 03-Oct-2008 19:06:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @view_spedges_OpeningFcn, ...
                   'gui_OutputFcn',  @view_spedges_OutputFcn, ...
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


% --- Executes just before view_spedges is made visible.
function view_spedges_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to view_spedges (see VARARGIN)

% Choose default command line output for view_spedges
handles.output = hObject;

handles = on_start(handles);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes view_spedges wait for user response (see UIRESUME)
% uiwait(handles.figure1);



function handles = on_start(handles)

% initial parameters
handles.files.filename = {'/osshare/Work/Data/nucleus_training_rotated/train/pos/nucleus00124.png' 'peppers.png' 'saturn.png'};
handles.files.pathname = {'' '' ''};
handles.sigma = 2;
%handles.angles = [0:45:360-45];
handles.angles = [0:30:360-30];

% create a window for the image and the overview panel
handles.hFig = figure('Toolbar','none','Menubar','none');
handles = load_image(handles,1); 
handles.hIm = imshow(handles.I);
handles.hSP = imscrollpanel(handles.hFig,handles.hIm);
set(handles.hSP,'Units','normalized','Position',[0 0 1 1])
handles.hov = imoverviewpanel(handles.overview,handles.hIm);

% set up the zoom slider
handles.zlist = [0.5 0.75 1 1.25 1.33 1.5 1.75 2 2.25 2.5 3 3.5 4 5 10 20];
zmin = 1;  zmax = length(handles.zlist);
slider_step(1) = 1/(zmax-zmin); slider_step(2) = 1/(zmax-zmin);
set(handles.zoom_slider,'sliderstep',slider_step, 'max', zmax,'min', zmin,'Value',5);

% set up the file listbox
set(handles.file_listbox, 'String', handles.files.filename);




function handles = load_image(handles, ind)

handles.I = imread([handles.files.pathname{ind} handles.files.filename{ind}]);

if size(handles.I,3) > 1
    handles.I = rgb2gray(handles.I);
end

handles.I = mat2gray(handles.I);

% compute the spedges
handles.sp = spedges(handles.I, handles.angles, handles.sigma, 11);
handles.EDGE = handles.sp.edge;

if get(handles.showedges, 'Value')  
    % get the edges image
    %handles.EDGE = edge(handles.I, 'log', 0, handles.sigma);
    % to handle directions that pass through lines at an angle, thicken lines
    % on diagonals
    %handles.EDGE = bwmorph(handles.EDGE, 'diag');   
    handles.I = imoverlay(handles.I, handles.EDGE, 'alpha', .5);
    handles.I = mat2gray(handles.I);
end





% --- Outputs from this function are returned to the command line.
function varargout = view_spedges_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function openfiles_Callback(hObject, eventdata, handles)
% hObject    handle to openfiles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

[filename, pathname, filterindex] = uigetfile( ...
       {'*.png','PNG Files (*.png)'; ...
        '*.jpg','JPEG Files (*.jpg)'; ...
        '*.*',  'All Files (*.*)'}, ...
        'Pick a file', ...
        'MultiSelect', 'on');
    
disp(filename)
disp(pathname)
disp(filterindex)


function file_listbox_Callback(hObject, eventdata, handles)
% hObject    handle to file_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns file_listbox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from file_listbox

contents = get(hObject,'String');
selected = contents{get(hObject,'Value')};
disp(['loading ' num2str(selected) ]);

ind = find(strcmp(handles.files.filename, selected));
handles = load_image(handles,ind); 


% display the new image in the image figure and the overview panel
 set(handles.hIm, 'CData', handles.I);                                       % update Image data
 ch = get(handles.overview, 'Children');chch = get(ch, 'Children');           % update Imoverviewpanel data
 chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);


%handles.hIm = imshow(handles.I);
%handles.hSP = imscrollpanel(handles.hFig,handles.hIm);
%handles.hov = imoverviewpanel(handles.overview,handles.hIm);

%get(handles.hIm)
disp('----------------');
%get(chch)
disp('----------------');
%get(chchch(2))

set(chch, 'YLim',  [0.5 size(handles.I,1)+.5]);
set(chch, 'XLim',  [0.5 size(handles.I,2)+.5]);

%handles.hIm = imshow(handles.I);
%handles.hSP = imscrollpanel(handles.hFig,handles.hIm);
%disp(contents);
%disp(selected);



function file_listbox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to file_listbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function showedges_Callback(hObject, eventdata, handles)
% hObject    handle to showedges (see GCBO)
% eventdcontents = get(hObject,'String');

contents = get(handles.file_listbox,'String');
selected = contents{get(handles.file_listbox,'Value')};
disp(['loading ' num2str(selected) ]);
ind = find(strcmp(handles.files.filename, selected));

handles = load_image(handles, ind);
%figure; imshow(handles.I);
%handles.I(1:10,1:10,1:3)
set(handles.hIm, 'CData', handles.I);                                   % update image data
ch = get(handles.overview, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);
guidata(hObject, handles);




function selectpoint_Callback(hObject, eventdata, handles)
% hObject    handle to selectpoint (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figure(handles.hFig);
[x, y] = ginput(1);
x = round(x);  y = round(y);
x = repmat(x, [length(handles.sp.angle) 1]);
y = repmat(y, [length(handles.sp.angle) 1]);

disp(['selected point x = ' num2str(x(1)) '; y = ' num2str(y(1)) ';']);

%handles.sp
sigma = 1;

sp1 = handles.sp.spedges(:,sigma,y(1),x(1)); sp1=squeeze(sp1); %bar(sp1);

for i = 1:length(handles.sp.angle)
    u(i) = sp1(i) * cosd(handles.sp.angle(i));
    v(i) = sp1(i) * -sind(handles.sp.angle(i));
end

%[handles.sp.angle' sp1 u' v']
hold on;
%quiver(x,y,u',v');
%quiver(x,y,u',v', 0);  % add the 0 to make the quivers NOT scaled down
quiver(x,y,u',v', 0, 'LineWidth', 2, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b', 'Marker', 'o', 'Color', 'b');
hold off;

axes(handles.axes1);
bar(sp1);
set(handles.axes1, 'XTickLabel', handles.sp.angle);



% x = repmat(a(1), [8 1])
% y = repmat(a(2), [8 1])
% 
% for i = 1:8
%     u(i) = sp1(i) * cosd(ang(i));
%     v(i) = sp1(i) * -sind(ang(i));
% end
% quiver(x,y,u',v')



function zoom_slider_Callback(hObject, eventdata, handles)
% hObject    handle to zoom_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

%disp(['zoom = ' num2str(get(hObject,'Value'))]);
roundedz = round(get(handles.zoom_slider, 'Value')); set(hObject, 'Value', roundedz);
z = handles.zlist(roundedz); set(handles.hz, 'String', ['zoom = ' num2str(z)]);
api = iptgetapi(handles.hSP);                                               % get API for the scrollpanel

mag = z; 
api = iptgetapi(handles.hSP);                                               % get API for the scrollpanel
r = api.getVisibleImageRect();                                              % find the visible rectangle for the scrollpanel
api.setMagnificationAndCenter(mag,r(1) + r(3)/2, r(2) + r(4)/2)    % set the magnification and center of scrollpanel

guidata(hObject, handles);  


function zoom_slider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zoom_slider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
delete(handles.hFig);
delete(hObject);


