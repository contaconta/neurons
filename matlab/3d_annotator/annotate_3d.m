function varargout = annotate_3d(varargin)
% annotate_3d M-file for annotate_3d.fig
%      annotate_3d, by itself, creates a new annotate_3d or raises the existing
%      singleton*.
%
%      H = annotate_3d returns the handle to a new annotate_3d or the handle to
%      the existing singleton*.
%
%      annotate_3d('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in annotate_3d.M with the given input arguments.
%
%      annotate_3d('Property','Value',...) creates a new annotate_3d or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before annotate_3d_OpeningFunction gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to annotate_3d_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help annotate_3d

% Last Modified by GUIDE v2.5 13-Aug-2009 23:42:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @annotate_3d_OpeningFcn, ...
                   'gui_OutputFcn',  @annotate_3d_OutputFcn, ...
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
%==========================================================================
%==========================================================================
% End initialization code - DO NOT EDIT
%==========================================================================
%==========================================================================


% --- Executes just before annotate_3d is made visible.
function annotate_3d_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to annotate_3d (see VARARGIN)

% Choose default command line output for annotate_3d
handles.output = hObject;
handles.GT.s = [];  handles.lines = []; handles.texts = []; handles.selected = []; handles.vlines = [ ]; handles.free = {}; handles.copied = [];
handles = load_database(handles);
filenm = [handles.overlayfolder handles.overlayprefix number_into_string(1,10) handles.suffix];
handles.oldGT = rgb2gray(imread(filenm)) > 0;
set(handles.hFig,'KeyPressFcn',{@keypresshandler, handles});       % key press handler
guidata(hObject, handles);                              % Update handles structure

% UIWAIT makes annotate_3d wait for user response (see UIRESUME)
% uiwait(handles.control_figure);


% --- Outputs from this function are returned to the command line.
function varargout = annotate_3d_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Get default command line output from handles structure
varargout{1} = handles.output;




% --- Executes when user attempts to close control_figure.
function control_figure_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to control_figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: delete(hObject) closes the figure
delete(hObject);

close all;


% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

% evaluate and update the slider values
roundeds = round(get(handles.slider1, 'Value')); set(hObject, 'Value', roundeds);

handles = grab_image(handles);                                              % load the image
set(handles.hs, 'String', ['s = ' num2str(roundeds)]);                      % update slider string
set(handles.hIm, 'CData', handles.I);                                       % update Image data
ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');           % update Imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);
guidata(hObject, handles);                                                  % store data to the gui                         

% THIS IS THE SUPPORTED WAY TO UPDATE THE FIGURE - BUT REFRESHING THE
% IMOVERVIEW MADE THIS INCREDIBLY SLOW, I HAVE HACKED IT BELOW
%---------------------------------------------------------------------
% api = iptgetapi(handles.hSP);
% r = api.getVisibleImageRect();
% mag = api.getMagnification();
% api.replaceImage(handles.I)
% api.setMagnificationAndCenter(mag,r(1) + r(3)/2, r(2) + r(4)/2)



% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

% evaluate and update the slider values
roundedt = round(get(handles.slider2, 'Value')); roundeds = round(get(handles.slider1, 'Value')); set(hObject, 'Value', roundedt);
if roundedt > length(handles.GT); handles.GT(roundedt).s = [ ]; end;


if get(handles.hgtoverlay, 'Value')
    filenm = [handles.overlayfolder handles.overlayprefix number_into_string(roundedt,10) handles.suffix];
    handles.oldGT = rgb2gray(imread(filenm)) > 0;
end

handles = grab_image(handles);                                          % load the image

set(handles.ht, 'String', ['t = ' num2str(roundedt)]);                  % update slider string
set(handles.hIm, 'CData', handles.I);                                   % update image data
ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);

if get(handles.hannotatetool, 'Value')
    set(handles.hselected, 'String', '[ no selection ]');
    api = iptgetapi(handles.hannotaterect);
    api.setColor([0 0 1]);
    handles.selected = [ ];
    set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
end
    

    
guidata(hObject, handles);                                              % store data to the gui

% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end




% --- Executes on button press in refreshoverview.
function refreshoverview_Callback(hObject, eventdata, handles)
% hObject    handle to refreshoverview (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles

get(handles.hlabel)


% --- Executes on slider movement.
function slider3_Callback(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

% update slider value and slider string for the ZOOM
roundedz = round(get(handles.slider3, 'Value')); set(hObject, 'Value', roundedz);
z = handles.zlist(roundedz); set(handles.hz, 'String', ['z=' num2str(z)]);


api = iptgetapi(handles.hSP);                                               % get API for the scrollpanel
r = api.getVisibleImageRect();                                              % find the visible rectangle for the scrollpanel
if get(handles.hannotatetool, 'Value')
    mag = z;
    api = iptgetapi(handles.hannotaterect);
    r = api.getPosition();
    api = iptgetapi(handles.hSP);
    api.setMagnificationAndCenter(mag,r(1) + r(3)/2, r(2) + r(4)/2)
else
    mag = z; 
    api = iptgetapi(handles.hSP);                                               % get API for the scrollpanel
    r = api.getVisibleImageRect();                                              % find the visible rectangle for the scrollpanel
    api.setMagnificationAndCenter(mag,r(1) + r(3)/2, r(2) + r(4)/2)    % set the magnification and center of scrollpanel
end
guidata(hObject, handles);                                                  % store data to the gui



% --- Executes during object creation, after setting all properties.
function slider3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



function hfolder_Callback(hObject, eventdata, handles)
% hObject    handle to hfolder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'String') returns contents of hfolder as text
%        str2double(get(hObject,'String')) returns contents of hfolder as a double

handles = load_database(handles);
disp(['Loaded database from ' get(handles.hfolder,'String')]);
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function hfolder_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hfolder (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on button press in hload.
function hload_Callback(hObject, eventdata, handles)
% hObject    handle to hload (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles = load_database(handles);
disp(['Loaded database from ' get(handles.hfolder,'String')]);
guidata(hObject, handles);



% --- Executes on button press in hinvert.
function hinvert_Callback(hObject, eventdata, handles)
% hObject    handle to hinvert (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of hinvert

handles = grab_image(handles);
set(handles.hIm, 'CData', handles.I);                                   % update image data
ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);
guidata(hObject, handles);



% --- Executes on button press in hgtoverlay.
function hgtoverlay_Callback(hObject, eventdata, handles)
% hObject    handle to hgtoverlay (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of hgtoverlay

roundedt = round(get(handles.slider2, 'Value'));

if get(handles.hgtoverlay, 'Value')
    filenm = [handles.overlayfolder handles.overlayprefix number_into_string(roundedt,10) handles.suffix];
    handles.oldGT = rgb2gray(imread(filenm)) > 0;
end

handles = grab_image(handles);
set(handles.hIm, 'CData', handles.I);                                   % update image data
ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);
guidata(hObject, handles);




% --- Executes on button press in hmip.
function hmip_Callback(hObject, eventdata, handles)
% hObject    handle to hmip (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of hmip
handles = grab_image(handles);
set(handles.hIm, 'CData', handles.I);                                   % update image data
ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);
guidata(hObject, handles);

% --- Executes on slider movement.
function hcontrast_Callback(hObject, eventdata, handles)
% hObject    handle to hcontrast (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider

handles.contrast = 1; handles = grab_image(handles);
roundedc = round(get(handles.hcontrast, 'Value')); set(hObject, 'Value', roundedc);
c = handles.clist(roundedc); set(handles.hc, 'String', ['Contrast c=' num2str(c,'%2.2f')]);
handles.contrast = c;
handles.I = imadjust(handles.I, [0; 1], [0; 1], c);
set(handles.hIm, 'CData', handles.I);                                   % update image data
ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);

guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function hcontrast_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hcontrast (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on button press in hlabel.
function hlabel_Callback(hObject, eventdata, handles)
% hObject    handle to hlabel (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



if get(handles.hannotatetool, 'Value')
    t = round(get(handles.slider2, 'Value')); s = round(get(handles.slider1, 'Value'));
    api = iptgetapi(handles.hannotaterect);
    r = api.getPosition();
    z = [s s];
    bb = round([r(1) r(2) z(1) r(3) r(4) z(2)-z(1)]);
    handles.undo = handles.GT;
    
    % if nothing is selected, we need to add a new label
    if isempty(handles.selected)   
        
        matchfound = 0;
        % check to see if this label matches a label on another slice
        if isfield(handles.GT, 's') && (length(handles.GT) >= t)
            for i = 1:length(handles.GT(t).s)
                if isequal([handles.GT(t).s(i).BoundingBox(1:2) handles.GT(t).s(i).BoundingBox(4:5)], [bb(1:2) bb(4:5)])
                    disp('match found on another layer!');
                    matchfound = i;
                end
            end
        end
            
        if matchfound > 0
            i = matchfound;
            z = min(s, handles.GT(t).s(matchfound).BoundingBox(3));
            z2 = max(handles.GT(t).s(matchfound).BoundingBox(3) + handles.GT(t).s(matchfound).BoundingBox(6), s);
            d = z2 - z;  
        elseif isfield(handles.GT, 's') && (length(handles.GT) >= t)
            i = length(handles.GT(t).s) + 1;
            d = 0; z = s;
        else
            i = 1;
            d = 0; z = s;
        end
        
        handles.GT(t).s(i).BoundingBox = round([bb(1:2) z bb(4:5) d]);
        handles.GT(t).s(i).color = 'b';
        handles = show_annotations(handles);
        pos = [handles.GT(t).s(i).BoundingBox(1:2)  handles.GT(t).s(i).BoundingBox(4:5)] - [.5 .5 0 0];           
        api.setPosition(pos);
        api.setColor([0 0 1]);
        set(handles.hselected, 'String', '[ no selection ]'); handles.selected = [ ]; set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
        disp(['Created annotation [t = ' num2str(t) '  i = ' num2str(i) ']']);
        
    % if something is already select, we need to adjust it
    else
        t = handles.selected(1);  i = handles.selected(2); s = round(get(handles.slider1, 'Value'));
        api = iptgetapi(handles.hannotaterect); r = api.getPosition();  
        x = r(1);  w = r(3);  y = r(2); h = r(4); 
        z = min(s, handles.GT(t).s(i).BoundingBox(3));
        z2 = max(handles.GT(t).s(i).BoundingBox(3) + handles.GT(t).s(i).BoundingBox(6), s);
        d = z2 - z;   % + 1???
        handles.GT(t).s(i).BoundingBox = round([x y z w h d]);
        handles.GT(t).s(i).color = 'b';
        handles = show_annotations(handles);
        api.setPosition([handles.GT(t).s(i).BoundingBox(1:2) - [.5 .5] handles.GT(t).s(i).BoundingBox(4:5)]);
        api.setColor([0 0 1]);
        set(handles.hselected, 'String', '[ no selection ]'); handles.selected = [ ]; set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
        disp(['Edited annotation [t = ' num2str(t) '  i = ' num2str(i) ']']);
    end

else
    disp('Enable the annotation tool before labeling.');
end
guidata(hObject, handles);



%-------------------------------------------
% SELECT
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

found = 0;
if get(handles.hannotatetool, 'Value')
    t = round(get(handles.slider2, 'Value')); s = round(get(handles.slider1, 'Value'));
    if isfield(handles.GT, 's') && (length(handles.GT) >= t)
        figure(handles.hFig);
        api = iptgetapi(handles.hSP);  r = api.getVisibleImageRect();
        htext = text(r(1)+r(3)/2, r(2) + r(4)/2, 'Click inside an annotation to select it');
        [x, y] = ginput(1);  delete(htext);
        for i = 1:length(handles.GT(t).s)
            zmin = handles.GT(t).s(i).BoundingBox(3);
            zmax = zmin + handles.GT(t).s(i).BoundingBox(6);


            if (s >= zmin - handles.WINDOW_SIZE) ...
                    && (s <= zmax + handles.WINDOW_SIZE) ...
                    && (x >= handles.GT(t).s(i).BoundingBox(1)) ...
                    && (x <= handles.GT(t).s(i).BoundingBox(1) + handles.GT(t).s(i).BoundingBox(4)) ...
                    && (y >= handles.GT(t).s(i).BoundingBox(2)) ...
                    && (y <= handles.GT(t).s(i).BoundingBox(2) + handles.GT(t).s(i).BoundingBox(5))
                    
                    handles.GT(t).s(i).color = 'b';
                    %get(handles.hannotaterect)
                    api = iptgetapi(handles.hannotaterect);
                    delete(handles.hannotaterect);
                    pos = [handles.GT(t).s(i).BoundingBox(1:2)  handles.GT(t).s(i).BoundingBox(4:5)] - [.5 .5 0 0];
                    handles.hannotaterect = imrect(imgca, pos);
                    %api.setPosition([handles.GT(t).s(i).BoundingBox(1:2) handles.GT(t).s(i).BoundingBox(4:5)]);
                    handles = show_annotations(handles);
                    api = iptgetapi(handles.hannotaterect);
                    api.setColor([1 0 0]);
                    handles.selected = [t i]; set(handles.hselected, 'BackgroundColor', [.75 0 0 ]);
                    disp(['Selected annotation [t = ' num2str(t) '  i = ' num2str(i) ']']);
                    set(handles.hselected, 'String', ['[t = ' num2str(t) '  i = ' num2str(i) ']']);
                    found = 1;
                    disp('----------------selected information---------------');
                    handles.GT(t).s(i)
                    disp('---------------------------------------------------');
                    
                    if get(handles.h3dcheck, 'Value')
                        figure(handles.h3d);
                        handles = volrender(handles);
                    end
                    
            end
        end
    end
    if ~found
        disp(['Could not find an annotation at [' num2str(x) ' ' num2str(y) ']']);
        set(handles.hselected, 'String', '[ no selection ]'); set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
        handles.selected = [ ];
        api = iptgetapi(handles.hannotaterect);
        position = api.getPosition();
        api.setColor([0 0 1]);
        api.setPosition([x-position(3)/2  y-position(4)/2 position(3) position(4)]);
    end   
else
    disp('Enable the annotation tool before labeling.');
end


guidata(hObject, handles);



% --- Executes on button press in hannotatetool.
function hannotatetool_Callback(hObject, eventdata, handles)
% hObject    handle to hannotatetool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of hannotatetool

if get(hObject,'Value')
    set(handles.hselected, 'String', '[ no selection ]'); set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
    handles.selected = [ ]; 
    api = iptgetapi(handles.hSP);                                               % get API for the scrollpanel
    r = api.getVisibleImageRect();                                              % find the visible rectangle for the scrollpanel
    handles.hannotaterect = imrect(imgca, [r(1)+r(3)*.4 r(2)+r(4)*.4   r(3)/10 r(4)/10]);
    api = iptgetapi(handles.hannotaterect);
    api.setColor([0 0 1]);
    set(handles.hlabel, 'Visible', 'on');
    set(handles.pushbutton2, 'Visible', 'on');
    set(handles.hremove, 'Visible', 'on');
    set(handles.h3dcheck, 'Visible', 'on');
    set(handles.hundo, 'Visible', 'on');
    set(handles.copy, 'Visible', 'on');
    set(handles.paste, 'Visible', 'on');

    %api.addNewPositionCallback(@(p)  api.setColor([0 0 1]) );    
else
    if isfield(handles, 'hannotaterect')
        if ~isempty(handles.hannotaterect)
            api = iptgetapi(handles.hannotaterect);
            %keyboard;
            %api.removeNewPositionCallback(handles.htoolmove)
            %api.delete(handles.hannotaterect);
            delete(handles.hannotaterect);
            handles.hannotaterect = [ ];
            set(handles.hselected, 'String', '[ no selection ]'); set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
            handles.selected = [ ];
            set(handles.hlabel, 'Visible', 'off');
            set(handles.pushbutton2, 'Visible', 'off');
            set(handles.hremove, 'Visible', 'off');
            set(handles.h3dcheck, 'Visible', 'off');
            set(handles.hundo, 'Visible', 'off');
            set(handles.copy, 'Visible', 'off');
            set(handles.paste, 'Visible', 'off');
        end
    end
end
guidata(hObject, handles);


% --- Executes on button press in hremove.
function hremove_Callback(hObject, eventdata, handles)
% hObject    handle to hremove (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isempty(handles.selected)
    t = handles.selected(1);  i = handles.selected(2);
    slist = 1:length(handles.GT(t).s);  slist = setdiff(slist, i);
    handles.GT(t).s = handles.GT(t).s(slist);
    set(handles.hselected, 'String', '[ no selection ]');
    set(handles.hselected, 'BackgroundColor', [.702 .702 .702]);
    handles.selected = [ ];
    handles = show_annotations(handles);
    api = iptgetapi(handles.hannotaterect);
    api.setColor([0 0 1]);
    set(handles.hIm, 'CData', handles.I);                                   % update image data
    ch = get(handles.ovpanel, 'Children');chch = get(ch, 'Children');       % update imoverviewpanel data
    chchch = get(chch, 'Children'); set(chchch(2), 'CData', handles.I);
    display(['Removed annotation [t = ' num2str(t) '  i = ' num2str(i) ']' ]);
    guidata(hObject, handles);
else
    display('No annotation selected.');
end
    

function hwindowsize_Callback(hObject, eventdata, handles)
% hObject    handle to hwindowsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hwindowsize as text
%        str2double(get(hObject,'String')) returns contents of hwindowsize as a double

handles.WINDOW_SIZE = str2double(get(hObject,'String'));
handles.WINDOW_SIZE
handles = show_annotations(handles);
guidata(hObject, handles);



% --- Executes during object creation, after setting all properties.
function hwindowsize_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hwindowsize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function hgtfilenm_Callback(hObject, eventdata, handles)
% hObject    handle to hgtfilenm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of hgtfilenm as text
%        str2double(get(hObject,'String')) returns contents of hgtfilenm as a double


% --- Executes during object creation, after setting all properties.
function hgtfilenm_CreateFcn(hObject, eventdata, handles)
% hObject    handle to hgtfilenm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in hloadgt.
function hloadgt_Callback(hObject, eventdata, handles)
% hObject    handle to hloadgt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(strfind(get(handles.hgtfilenm, 'String') , '/'));
    folder = [pwd '/'];
    filenm = [folder get(handles.hgtfilenm, 'String')];
else
    filenm = get(handles.hgtfilenm, 'String');
end

try
    load(filenm);
    handles.GT = GT;
    handles.undo = GT;
    disp(['Loaded annotations from file: ' filenm ]);
catch
    disp(['The file you have requested ' filenm ' could not be loaded or does not exist.']);
end

handles = show_annotations(handles);
guidata(hObject, handles);

% --- Executes on button press in hsavegt.
function hsavegt_Callback(hObject, eventdata, handles)
% hObject    handle to hsavegt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isempty(strfind(get(handles.hgtfilenm, 'String') , '/'));
    folder = [pwd '/'];
    filenm = [folder get(handles.hgtfilenm, 'String')];
else
    filenm = get(handles.hgtfilenm, 'String');
end

cmd = ['cp ' filenm ' ' filenm '~'];
[status, result] = system(cmd);
if status == 0
    disp([cmd ' successful.']);
else
    disp([cmd ' unsuccessful.']);
end
GT = handles.GT;
save(filenm, 'GT');
disp(['Wrote annotations to file: ' filenm ]);

% --- Executes on button press in h3dcheck.
function h3dcheck_Callback(hObject, eventdata, handles)
% hObject    handle to h3dcheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hint: get(hObject,'Value') returns toggle state of h3dcheck

% IF VALUE = 1
if get(hObject,'Value')
    handles.h3d = figure;
    %figure(handles.h3d);
    handles = volrender(handles);
else
    if ~isempty(handles.vlines)
        for i = 1:length(handles.vlines);  delete(handles.vlines(i)); end
        handles.vlines = [ ];
    end
    delete(handles.h3d);
    handles.h3d = [];
end
guidata(hObject, handles);


% --- Executes on button press in freedraw.
function freedraw_Callback(hObject, eventdata, handles)
% hObject    handle to freedraw (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%handles.free{length(handles.free + 1)} = imfreehand(imgca);
%h = imfreehand(imgca, 'Closed', 0);
h = imfreehand(imgca);
l = get(h, 'Children');
for i=1:length(l)-1
    set(l(i), 'Color', [0 1 0]);
end
handles.free{length(handles.free) + 1 } = h;
guidata(hObject, handles);

% --- Executes on button press in freeclear.
function freeclear_Callback(hObject, eventdata, handles)
% hObject    handle to freeclear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isempty( handles.free)
    delete(handles.free{length(handles.free)});
    handles.free = handles.free( 1:length(handles.free) - 1);
else
    handles.free = {};
end

%for i = 1:length(handles.free); delete(handles.free{i}); end
%handles.free = { }; 
guidata(hObject, handles);

% --- Executes on button press in freehand.
function freehand_Callback(hObject, eventdata, handles)
% hObject    handle to freehand (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%disp([ 'handles.free: [' num2str(handles.free)  ']'])

if get(hObject,'Value')
    for i = 1:length(handles.free)
        %get(handles.free{i})
        set(handles.free{i}, 'Visible', 'on');
    end
        
else
    for i = 1:length(handles.free)
        set(handles.free{i}, 'Visible', 'off');
    end
end
guidata(hObject, handles);


% --- Executes on button press in hundo.
function hundo_Callback(hObject, eventdata, handles)
% hObject    handle to hundo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.GT = handles.undo;
handles = show_annotations(handles);
guidata(hObject, handles);

%==========================================================================
% --- Function to grab the correct image from the database
%==========================================================================
function handles = grab_image(handles)

% define the filename
roundedt = round(get(handles.slider2, 'Value')); roundeds = round(get(handles.slider1, 'Value'));
filenm = [ handles.folder handles.prefix number_into_string(roundedt,10) handles.midfix number_into_string(roundeds,10) handles.suffix ];

if get(handles.hmip, 'Value')  
    % get a maximum intensity projection
    filenm = [ handles.overlayfolder 'denoised' number_into_string(roundedt,10) handles.suffix ];
    handles.I = imread(filenm);
else
    % get the image slice defined by 'roundeds'
    filenm = [ handles.folder handles.prefix number_into_string(roundedt,10) handles.midfix number_into_string(roundeds,10) handles.suffix ];
    handles.I = imread(filenm);
end

% check to see if we need to invert the color 
if ~get(handles.hinvert, 'Value')   
    handles.I = imcomplement(handles.I);
end

% adjust the contrast if necessary
if handles.contrast ~= 1
    handles.I = imadjust(handles.I, [0; 1], [0; 1], handles.contrast);
end

% check to see if we need to add the GToverlay
if get(handles.hgtoverlay, 'Value')
    handles.I = imoverlay(handles.I, handles.oldGT, 'bright', 'color', [0 1 0], 'alpha', .7);
end

handles = show_annotations(handles);

disp(['loaded ' filenm]);



%==========================================================================
% --- Function to load the database and set up the sliders, etc
%==========================================================================
function handles = show_annotations(handles)

t = round(get(handles.slider2, 'Value')); s = round(get(handles.slider1, 'Value'));

% first, remove all previous lines
for i = 1:length(handles.lines);  delete(handles.lines(i)); end
for i = 1:length(handles.texts);  delete(handles.texts(i)); end
handles.lines = [ ]; 
handles.texts = [ ];

figure(handles.hFig);
if isfield(handles.GT, 's') && (length(handles.GT) >= t)
    for i = 1:length(handles.GT(t).s)
        zmin = handles.GT(t).s(i).BoundingBox(3);
        zmax = zmin + handles.GT(t).s(i).BoundingBox(6);
        
        
        if s >= zmin && s <= zmax
            
            x1 = handles.GT(t).s(i).BoundingBox(1)  - .5;
            y1 = handles.GT(t).s(i).BoundingBox(2)  - .5;
            x2 = x1 + handles.GT(t).s(i).BoundingBox(4);
            y2 = y1 + handles.GT(t).s(i).BoundingBox(5);

            handles.lines(length(handles.lines)+1) = line([x1 x2], [y1 y1]); set(handles.lines(length(handles.lines)), 'LineWidth', 2, 'Color', handles.GT(t).s(i).color);
            handles.lines(length(handles.lines)+1) = line([x2 x2], [y1 y2]); set(handles.lines(length(handles.lines)), 'LineWidth', 2, 'Color', handles.GT(t).s(i).color);
            handles.lines(length(handles.lines)+1) = line([x2 x1], [y2 y2]); set(handles.lines(length(handles.lines)), 'LineWidth', 2, 'Color', handles.GT(t).s(i).color);
            handles.lines(length(handles.lines)+1) = line([x1 x1], [y2 y1]); set(handles.lines(length(handles.lines)), 'LineWidth', 2, 'Color', handles.GT(t).s(i).color);
            handles.texts(length(handles.texts)+1) = text(x1, y1, ['t=' num2str(t) ' i=' num2str(i) ]);
            
        elseif (s + handles.WINDOW_SIZE >= zmin  && s < zmin) || (s - handles.WINDOW_SIZE <= zmax  && s > zmax)
            x1 = handles.GT(t).s(i).BoundingBox(1)  - .5;
            y1 = handles.GT(t).s(i).BoundingBox(2)  - .5;
            x2 = x1 + handles.GT(t).s(i).BoundingBox(4);
            y2 = y1 + handles.GT(t).s(i).BoundingBox(5);

            handles.lines(length(handles.lines)+1) = line([x1 x2], [y1 y1]); set(handles.lines(length(handles.lines)), 'LineStyle', '-.', 'Color', handles.GT(t).s(i).color);
            handles.lines(length(handles.lines)+1) = line([x2 x2], [y1 y2]); set(handles.lines(length(handles.lines)), 'LineStyle', '-.', 'Color', handles.GT(t).s(i).color);
            handles.lines(length(handles.lines)+1) = line([x2 x1], [y2 y2]); set(handles.lines(length(handles.lines)), 'LineStyle', '-.', 'Color', handles.GT(t).s(i).color);
            handles.lines(length(handles.lines)+1) = line([x1 x1], [y2 y1]); set(handles.lines(length(handles.lines)), 'LineStyle', '-.', 'Color', handles.GT(t).s(i).color);
        end
    end
end



%==========================================================================
% --- Function to load the database and set up the sliders, etc
%==========================================================================
function handles = load_database(handles)

if isfield(handles, 'hFig')
    delete(handles.hFig);           % if something is already loaded, delete the image figure
    delete(handles.hov);            % if something is already loaded, delete the imoverviewpanel
end

smin = 1; smax = 31;
slider_step(1) = 1/(smax-smin); slider_step(2) = 1/(smax-smin);
set(handles.slider1,'sliderstep',slider_step, 'max', smax,'min', smin,'Value',1);

tmin = 1; tmax = 24;
slider_step(1) = 1/(tmax-tmin); slider_step(2) = 1/(tmax-tmin);
set(handles.slider2,'sliderstep',slider_step, 'max', tmax,'min', tmin,'Value',1);

%handles.folder = '/osshare/Work/Data/060824-S1-26-Stacks/preprocessed/';
handles.folder = '/osshare/Work/Data/060824-S1-26-stabilized/';
set(handles.hfolder, 'String', handles.folder);
handles.prefix =  'preprocess_t'; 
handles.midfix =  '_s'; handles.suffix =  '.png';
handles.overlayfolder = '/osshare/Work/Data/tracking_2d_data/';
handles.overlayprefix = 'GT';

set(handles.hsmin, 'String', num2str(smin)); set(handles.hsmax, 'String', num2str(smax));
set(handles.htmin, 'String', num2str(tmin)); set(handles.htmax, 'String', num2str(tmax));

handles.zlist = [0.5 0.75 1 1.25 1.33 1.5 1.75 2 2.25 2.5 3 3.5 4 5 10 20];
zmin = 1;  zmax = length(handles.zlist);
slider_step(1) = 1/(zmax-zmin); slider_step(2) = 1/(zmax-zmin);
set(handles.slider3,'sliderstep',slider_step, 'max', zmax,'min', zmin,'Value',5);

handles.zoom = handles.zlist(get(handles.slider3, 'Value'));
set(handles.hzmin, 'String', num2str(handles.zlist(1)));
set(handles.hzmax, 'String', num2str(handles.zlist(length(handles.zlist))));
set(handles.hz, 'String', ['z=' num2str(handles.zoom)]);
set(handles.hwindowsize, 'String', '31');  handles.WINDOW_SIZE = 31;

handles.clist = [1/20 1/18 1/16 1/14 1/12 1/10 1/9 1/8 1/7 1/6 1/5 1/4 1/3.5 1/3 1/2.5 1/2 1/1.75 1/1.5 1/1.33 1/1.25 1 1.1 1.25 1.33 1.5 1.75 2 2.25 2.5 3 3.5 4 5 6 7 8 9 10 12 14 16 18 20];
cmin = 1;  cmax = length(handles.clist);
slider_step(1) = 1/(cmax-cmin);
slider_step(2) = 1/(cmax-cmin);
set(handles.hcontrast,'sliderstep',slider_step, 'max', cmax,'min', cmin,'Value',21);
handles.contrast = handles.clist(get(handles.hcontrast, 'Value'));
set(handles.hc, 'String', ['Contrast c=' num2str(handles.contrast)]);

handles.hFig = figure('Toolbar','none','Menubar','none');
%handles.hFig = figure;
handles = grab_image(handles);                                          % load the image
handles.hIm = imshow(handles.I);
handles.hSP = imscrollpanel(handles.hFig,handles.hIm);
set(handles.hSP,'Units','normalized','Position',[0 0 1 1])

handles.hov = imoverviewpanel(handles.ovpanel,handles.hIm);

handles.WINDOW_SIZE = str2double(get(handles.hwindowsize, 'String'));
set(handles.hlabel, 'Visible', 'off');
set(handles.pushbutton2, 'Visible', 'off');
set(handles.hremove, 'Visible', 'off');
set(handles.h3dcheck, 'Visible', 'off');
set(handles.hundo, 'Visible', 'off');

% for the magnification box, if I wanted it!
%set(handles.hSP,'Units','normalized','Position',[0 .1 1 .9]) 
%handles.hMagBox = immagbox(handles.hFig,handles.hIm);
%handles.pos = get(handles.hMagBox,'Position');
%set(handles.hMagBox,'Position',[0 0 handles.pos(3) handles.pos(4)]);


%==========================================================================
% --- Executes on a key press on hFig - the figure containing the image
%==========================================================================
% --- Executes on key press with focus on control_figure and no controls selected.
function keypresshandler(src, evnt, handles)

src = handles.control_figure;
handles = guidata(src);

switch evnt.Key
    case 'uparrow'
        svalue = min(str2double(get(handles.hsmax, 'String')), get(handles.slider1, 'Value') + 1);
        set(handles.slider1, 'Value', svalue);
        slider1_Callback(handles.slider1, [ ], handles);
    case 'downarrow'
        svalue = max(str2double(get(handles.hsmin, 'String')), get(handles.slider1, 'Value') - 1);
        set(handles.slider1, 'Value', svalue);
        slider1_Callback(handles.slider1, [ ], handles);
    case 'rightarrow'
        tvalue = min(str2double(get(handles.htmax, 'String')), get(handles.slider2, 'Value') + 1);
        set(handles.slider2, 'Value', tvalue);
        slider2_Callback(handles.slider2, [ ], handles);
    case 'leftarrow'
        tvalue = max(str2double(get(handles.htmin, 'String')), get(handles.slider2, 'Value') - 1);
        set(handles.slider2, 'Value', tvalue);
        slider2_Callback(handles.slider2, [ ], handles);
    case 'm'
        set(handles.hmip, 'Value', ~get(handles.hmip, 'Value') );
        hmip_Callback(handles.hmip, [ ], handles);
    case 'period'
        zvalue = min(get(handles.slider3,'Max'), get(handles.slider3, 'Value') + 1);
        set(handles.slider3, 'Value', zvalue);
        slider3_Callback(handles.slider3, [ ], handles);
    case 'comma'
        zvalue = max(get(handles.slider3,'Min'), get(handles.slider3, 'Value') - 1);
        set(handles.slider3, 'Value', zvalue);
        slider3_Callback(handles.slider3, [ ], handles);
    case 'a'
        set(handles.hannotatetool, 'Value', ~get(handles.hannotatetool, 'Value'));
        hannotatetool_Callback(handles.hannotatetool, [ ], handles);
    case 'return'
        hlabel_Callback(handles.hlabel, [ ], handles);
    case 's'
        pushbutton2_Callback(handles.pushbutton2, [ ], handles);
    case 'r'
        hremove_Callback(handles.hremove, [ ], handles);
    case 'i'
        set(handles.hinvert, 'Value', ~get(handles.hinvert, 'Value'));
        hinvert_Callback(handles.hinvert, [ ], handles);
    case 'o'
        set(handles.hgtoverlay, 'Value', ~get(handles.hgtoverlay, 'Value'));
        hgtoverlay_Callback(handles.hgtoverlay, [ ], handles);
    case '3'
        set(handles.h3dcheck, 'Value', ~get(handles.h3dcheck, 'Value'));
        h3dcheck_Callback(handles.h3dcheck, [ ], handles);
    case 'u'
        hundo_Callback(handles.hundo, [ ], handles);
    case 'f'
        set(handles.freehand, 'Value', ~get(handles.freehand, 'Value'));
        freehand_Callback(handles.freehand, [ ], handles)
    case 'd'
        freedraw_Callback(handles.freedraw, [ ], handles);
    case 'e'
        freeclear_Callback(handles.freeclear, [ ], handles);   
    case 'c'
        copy_Callback(handles.copy, [ ] , handles);
    case 'p'
        paste_Callback(handles.paste, [ ] , handles);
    case 'h'
        disp('------keyboard shortcut help-------')
        disp('*       up and down arrows change the slice')
        disp('*       left and right arrows change the time step')
        disp('m       toggles the maximum intensity projection')
        disp('i       toggles invert colors');
        disp('o       toggles the old ground truth overlay');
        disp('.       increases the zoom level')
        disp(',       decreases the zoom level')
        disp('a       toggles the annotation tool on and off')
        disp('s       selects an annotation')
        disp('r       removes an annotation')
        disp('u       undo the last change to annotations');
        disp('3       shows a 3D volume of the selection');
        disp('enter   adds a new annotation or edits an existing annotation');
        disp('f       toggles freehand drawings on/off');
        disp('d       creates a new freehand drawing');
        disp('e       erases the last freehand drawing');
        disp('c       copies selected annotations');
        disp('p       pastes annotations');
        disp('-----------------------------------')
    otherwise
        disp('unrecognized key command');
end

disp([' you pressed "' evnt.Key '"']);
%guidata(src, handles);


%==========================================================================
% --- Volume render
%==========================================================================
function handles = volrender(handles)

if ~isempty(handles.selected)
    XYBUFFER = 13;  ZBUFFER = 3;
    if ~isempty(handles.vlines)
        for i = 1:length(handles.vlines);  delete(handles.vlines(i)); end
    end
    t = handles.selected(1);  i = handles.selected(2);
    x = handles.GT(t).s(i).BoundingBox(1);
    y = handles.GT(t).s(i).BoundingBox(2);
    z = handles.GT(t).s(i).BoundingBox(3);
    w = handles.GT(t).s(i).BoundingBox(4); 
    h = handles.GT(t).s(i).BoundingBox(5); 
    d = handles.GT(t).s(i).BoundingBox(6);
    zb1 = max(1,z - ZBUFFER);
    zb2 = min(str2double(get(handles.hsmax, 'String')), z + d + ZBUFFER);
    slist = [zb1:zb2];
    for ind = 1:length(slist)
        s = slist(ind);
        filenm = [ handles.folder handles.prefix number_into_string(t,10) handles.midfix number_into_string(s,10) handles.suffix ];
        I = mat2gray(imcomplement(imread(filenm))); %disp(['loaded ' filenm]);
        xb1 = max(1,x - XYBUFFER);
        yb1 = max(1,y - XYBUFFER);
        xb2 = min(size(I,2), x + w + XYBUFFER);
        yb2 = min(size(I,1), y + h + XYBUFFER);
        V(:,:,1,ind) = I(yb1:yb2, xb1:xb2);
        disp(['loaded ' filenm]);
    end
    disp(['loaded files needed for 3d rendering.']);
    V = squeeze(V);    vol3d('cdata',V,'texture','3D');
    view(3);     axis tight;  daspect([1 1 .3])
    colormap('gray'); alphamap('rampdown'); alphamap(.75 .* alphamap); grid on;
    xv1 = min(XYBUFFER, max(0, x - XYBUFFER));
    yv1 = min(XYBUFFER, max(0, y - XYBUFFER));
    %zv1 = min(ZBUFFER, max(0, z - ZBUFFER));
    if z <= ZBUFFER; zv1 = 0 - (z - ZBUFFER);  else zv1 = ZBUFFER; end
    xv2 = xv1 + w;  % +1 ?
    yv2 = yv1 + h;
    zv2 = zv1 + d + 1;
    xs = [xv1 xv2 xv2 xv1 xv1   xv1 xv2 xv2 xv1 xv1 ];  
    ys = [yv1 yv1 yv2 yv2 yv1   yv1 yv1 yv2 yv2 yv1 ];  
    zs = [zv1 zv1 zv1 zv1 zv1   zv2 zv2 zv2 zv2 zv2 ]; 
    handles.vlines(length(handles.vlines)+1) = line(xs, ys, zs);
    handles.vlines(length(handles.vlines)+1) = line([xv2 xv2], [yv1 yv1], [zv1 zv2]);
    handles.vlines(length(handles.vlines)+1) = line([xv1 xv1], [yv2 yv2], [zv1 zv2]);
    handles.vlines(length(handles.vlines)+1) = line([xv2 xv2], [yv2 yv2], [zv1 zv2]);
end


% --- Executes on button press in paste.
function paste_Callback(hObject, eventdata, handles)
% hObject    handle to paste (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
t = round(get(handles.slider2, 'Value'));

if isempty(handles.copied)
    disp('You must copy annotations before you can paste.');
    return;
end
if handles.copied(1,1) == t
    disp('You cannot copy and paste an annotation on the same time step');
    return;
end

if get(handles.hannotatetool, 'Value')
    handles.undo = handles.GT;
    
    for j = 1:size(handles.copied,1)
        oldt = handles.copied(j,1);
        oldi = handles.copied(j,2);
       
        if isfield(handles.GT, 's') && (length(handles.GT) >= t) && length(handles.GT(t).s) >= 1
            i = length(handles.GT(t).s) + 1;
            handles.GT(t).s(i).BoundingBox = handles.GT(oldt).s(oldi).BoundingBox;
            handles.GT(t).s(i).color = 'g';
        else
            i = 1;
            handles.GT(t).s(i).BoundingBox = handles.GT(oldt).s(oldi).BoundingBox;
            handles.GT(t).s(i).color = 'g';
        end
        
        disp(['pasted annotation t=' num2str(oldt)  ' s=' num2str(oldi) 'as t=' num2str(t) ' s=' num2str(j)]);
    end

end

handles = show_annotations(handles);
guidata(hObject, handles);

% --- Executes on button press in copy.
function copy_Callback(hObject, eventdata, handles)
% hObject    handle to copy (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
t = round(get(handles.slider2, 'Value')); s = round(get(handles.slider1, 'Value'));

api = iptgetapi(handles.hSP);                                               % get API for the scrollpanel
r = api.getVisibleImageRect();                                              % find the visible rectangle for the scrollpanel
handles.copyrect = imrect(imgca, [r(1)+r(3)*.4 r(2)+r(4)*.4   r(3)/10 r(4)/10]);
api = iptgetapi(handles.copyrect);
api.setColor([1 0 1]);

disp('Select the annotations you wish to copy using the magenta rectangle.');
disp('Double-click on the magenta rectangle when you are finished');
R = wait(handles.copyrect);
delete(handles.copyrect);
handles.copyrect = [ ];

% DO SOME LOGIC TO FIGURE OUT WHICH ANNOTATIONS WERE SELECTED - THEN PASS
% THEM TO HANDLES SO PASTE CAN BE USED 
handles.copied = [ ];
if isfield(handles.GT, 's') && (length(handles.GT) >= t)
    for i = 1:length(handles.GT(t).s)
        q(1:2) = handles.GT(t).s(i).BoundingBox(1:2);
        q(3:4) = handles.GT(t).s(i).BoundingBox(4:5);
        if isinsiderect(R, q) == 1
            handles.copied(size(handles.copied,1) + 1,:) = [t i]';
        end
    end
end
if ~isempty(handles.copied)
    disp(['copied annotations (t=' num2str(t) '): ' num2str(handles.copied(:,2)') ]);
else
    disp('No annotations found to copy.');
end
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function control_figure_CreateFcn(hObject, eventdata, handles)
% hObject    handle to control_figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object deletion, before destroying properties.
function control_figure_DeleteFcn(hObject, eventdata, handles)
% hObject    handle to control_figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
