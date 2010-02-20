function varargout = annotator3D(varargin)
% ANNOTATOR3D M-file for annotator3D.fig
%      ANNOTATOR3D, by itself, creates a new ANNOTATOR3D or raises the existing
%      singleton*.
%
%      H = ANNOTATOR3D returns the handle to a new ANNOTATOR3D or the handle to
%      the existing singleton*.
%
%      ANNOTATOR3D('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ANNOTATOR3D.M with the given input arguments.
%
%      ANNOTATOR3D('Property','Value',...) creates a new ANNOTATOR3D or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before annotator3D_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to annotator3D_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help annotator3D

% Last Modified by GUIDE v2.5 19-Feb-2010 14:48:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @annotator3D_OpeningFcn, ...
                   'gui_OutputFcn',  @annotator3D_OutputFcn, ...
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


% --- Executes just before annotator3D is made visible.
function annotator3D_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to annotator3D (see VARARGIN)

% Choose default command line output for annotator3D
handles.output = hObject;

handles.hFig = figure('Toolbar','none','Menubar','none');
handles.hIm = imshow(zeros(1500)); %imshow(imread('peppers.png'));
handles.hSP = imscrollpanel(handles.hFig,handles.hIm);
handles.api = iptgetapi(handles.hSP);
handles.hFigAxis = gca;
handles.selected = [];
handles.clipboard = [];
handles.IMSIZE = [];
handles.polys =[];
handles.hline = [];
handles.show = 1;
set(handles.hSP,'Units','normalized','Position',[0 0 1 1])
set(handles.hFig,'KeyPressFcn',{@keypresshandler, handles}); % set an event listener for keypresses
handles.hov = imoverviewpanel(handles.ovpanel, handles.hIm);


% INITIALIZE ALL HANDLES WE WILL NEED LATER!

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes annotator3D wait for user response (see UIRESUME)
% uiwait(handles.Annotator3D);

% --- Outputs from this function are returned to the command line.
function varargout = annotator3D_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;








%% =============== IMAGE SLICE FOLDER STUFF =============================
% --- Executes on button press in imagefolderbutton.
function imagefolderbutton_Callback(hObject, eventdata, handles)
% hObject    handle to imagefolderbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.imagefolder = [uigetdir(handles.imagefolder, 'Select a Directory') '/'];
set(handles.imagefoldertxt, 'String', handles.imagefolder);
guidata(hObject, handles);
handles = loadimagefolder(handles);
guidata(hObject, handles);

function imagefoldertxt_Callback(hObject, eventdata, handles) %#ok<*INUSD>
% hObject    handle to xmlfoldertxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.imagefolder = get(hObject, 'String');
guidata(hObject, handles);
handles = loadimagefolder(handles);
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function imagefoldertxt_CreateFcn(hObject, eventdata, handles)
% hObject    handle to imagefoldertxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.imagefolder = [pwd '/'];
set(hObject, 'String', handles.imagefolder);  % set the initial path to pwd on creation
guidata(hObject, handles);

% --- LOADS IMAGE SLICE FOLDER.
function handles = loadimagefolder(handles)
% build a list of valid files contained in the folder
searchstring = [handles.imagefolder get(handles.namefiltertxt, 'String')];%disp(searchstring);
handles.d_image = dir(searchstring);

if isempty(handles.d_image)
    handles.error = errordlg(['No image files matching ' get(handles.namefiltertxt, 'String') ' found in ' handles.imagefolder], 'Error: No matching files found.');
else    
    % load and display the first file in the list
    TEMP_FILENAME = handles.d_image(1).name;
    filename = [handles.imagefolder TEMP_FILENAME];
    %handles.I = imread([handles.imagefolder handles.d_image(1).name]);
    set(handles.hFig, 'Name', TEMP_FILENAME);
    handles.I = imread(filename);
    handles.IMSIZE = size(handles.I);
    handles.Ifilename = TEMP_FILENAME;
    handles.api.replaceImage(handles.I, 'PreserveView', 1);
    
    % set up our Z slider
    pat = '\d*';
    Zmin = str2double(regexp(handles.d_image(1).name,pat, 'match'));
    Zmax = str2double(regexp(handles.d_image(length(handles.d_image)).name,pat, 'match'));
    set(handles.zslider, 'Value', Zmin);
    set(handles.z, 'String', num2str(Zmin));
    %set(handles.zslider, 'min', Zmin);  
    set(handles.zmintxt, 'String', num2str(Zmin));
    %set(handles.zslider, 'max', Zmax);
    set(handles.zmaxtxt, 'String', num2str(Zmax));
    sliderstep(1) = 1/(Zmax - Zmin); sliderstep(2) = sliderstep(1);
    set(handles.zslider,'sliderstep',sliderstep, 'max', Zmax,'min', Zmin,'Value',Zmin);

    % delete the current annotations
    if ~isempty(handles.hline)
        for i = 1:length(handles.hline);
            delete(handles.hline{i});
        end
    end
    handles.hline = [];
    handles.polys = [];
    
    % check to see if we can load an annotation
    handles = checkIMloadXML(handles);
    
    %set(handles.hIm, 'CData', handles.I);%ch = get(handles.ovpanel, 'Children'); chch = get(ch, 'Children');%chchch = get(chch, 'Children'); set(chchch(2), 'CData', handle.I);
end








%% ================== XML FOLDER STUFF ================================
% --- Executes on button press in xmlfolderbutton.
function xmlfolderbutton_Callback(hObject, eventdata, handles)
% hObject    handle to xmlfolderbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.xmlfolder = [uigetdir(handles.xmlfolder, 'Select a Directory') '/'];
set(handles.xmlfoldertxt, 'String', handles.xmlfolder);
guidata(hObject, handles);
handles = loadxmlfolder(handles);
guidata(hObject, handles);

function xmlfoldertxt_Callback(hObject, eventdata, handles) %#ok<*INUSD>
% hObject    handle to xmlfoldertxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.xmlfolder = get(hObject, 'String');
guidata(hObject, handles);
handles = loadxmlfolder(handles);
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function xmlfoldertxt_CreateFcn(hObject, eventdata, handles) %#ok<*INUSL,*DEFNU>
% hObject    handle to xmlfoldertxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
handles.xmlfolder = [pwd '/'];
set(hObject, 'String', handles.xmlfolder);  % set the initial path to pwd on creation
guidata(hObject, handles);

% --- LOADS XML FOLDER.
function handles = loadxmlfolder(handles)
% build a list of valid files contained in the folder
searchstring = [handles.xmlfolder get(handles.xmlfilter, 'String')];
disp(searchstring);
handles.d_xml = dir(searchstring);

if isempty(handles.d_xml)
    handles.error = errordlg(['No XML files matching ' get(handles.namefiltertxt, 'String') ' found in ' handles.xmlfolder], 'Error: No matching files found.');
else    
    % load and display the annotation corresponding to the current image 
    handles = checkIMloadXML(handles);
end






% ================ SELECT BUTTON STUFF ================================
% --- Executes on button press in selectbutton.
function selectbutton_Callback(hObject, eventdata, handles)
% hObject    handle to selectbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

axes(handles.hFigAxis); %#ok<MAXES>
[x y] = myginput(1, 'circle');
%disp([' point ' num2str(x) ' ' num2str(y)]);

in = 0; p = 1;
while (in == 0) && (p <= length(handles.polys))
    in = inpoly([x y], handles.polys{p});
    if in == 1
        handles = deselectprevious(handles);
        handles = selectpoly(handles, p);
        break; 
    end
    if p == length(handles.polys)
        handles = deselectprevious(handles);
        disp('Could not find any polygon matching your selection');
    end
    p = p + 1;    
end
guidata(hObject, handles);

function handles = selectpoly(handles, p)

%disp(['selected poly ' num2str(p)]);
inds = setdiff(1:length(handles.polys), p);

delete(handles.hline{p});
pselected = handles.polys{p};

handles.hline = handles.hline(inds);
handles.polys = handles.polys(inds);
handles.selected = impoly(handles.hFigAxis, pselected);


function handles = deselectprevious(handles)
if ~isempty(handles.selected)
    pol = getPosition(handles.selected);
    delete(handles.selected);
    handles.selected = [];
    
    % add the old selected poly to the set of polys
    pol_ind = length(handles.polys)+1;
    handles.polys{pol_ind} = pol;
    
    
    % delete all shown annotations
    if ~isempty(handles.hline)
        for i = 1:length(handles.hline);
            delete(handles.hline{i});
        end
    end
    handles.hline = [];

    % now draw all the annotations
    polys = handles.polys;
    axes(handles.hFigAxis); %#ok<MAXES>
    hold on; 
    for p = 1:length(polys)
        pol = polys{p};
        handles.hline{p} = plot([pol(:,1); pol(1,1)],[pol(:,2); pol(1,2)],'r','LineWidth',2);
    end
    hold off;
    handles.show = 1;

%     if handles.show == 1
%         p = length(handles.hline) + 1;
%         axes(handles.hFigAxis); %#ok<MAXES>
%         hold on;
%         handles.hline{p} = plot([pol(:,1); pol(1,1)],[pol(:,2); pol(1,2)],'r','LineWidth',2);
%         hold off;
%     end

end

% --- Executes on button press in unselectbutton.
function unselectbutton_Callback(hObject, eventdata, handles)
% hObject    handle to unselectbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles = deselectprevious(handles);
guidata(hObject, handles);








% ================ ADD A NEW ANNOTATION ================================
% --- Executes on button press in newpolybutton.
function newpolybutton_Callback(hObject, eventdata, handles)
% hObject    handle to newpolybutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

disp('Creating a new annotation, click points to create a polygon.');
handles.selected = impoly(handles.hFigAxis);
guidata(hObject, handles);




% ================ DELETE AN ANNOTATION ================================
% --- Executes on button press in deletepoly.
function deletepoly_Callback(hObject, eventdata, handles)
% hObject    handle to deletepoly (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isempty(handles.selected)
    delete(handles.selected);
    handles.selected = [];
    disp('Deleted the selected polygon.');
    guidata(hObject, handles);
else
    disp('Error: No annotation selected. Could not delete.');
end



% ================ COPY AN ANNOTATION ================================
% --- Executes on button press in copybutton.
function copybutton_Callback(hObject, eventdata, handles)
% hObject    handle to copybutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if ~isempty(handles.selected)
    pol = getPosition(handles.selected);    
    handles.clipboard = pol;
    disp('Copied the annotation to the clipboard.');
end
guidata(hObject, handles);

% --- Executes on button press in copyallbutton.
function copyallbutton_Callback(hObject, eventdata, handles)
% hObject    handle to copyallbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if ~isempty(handles.polys)
    handles.clipboard = handles.polys;
    disp('Copied ALL annotations on this SLICE to the clipboard.');
end
guidata(hObject, handles);

% ================ PASTE AN ANNOTATION ================================
% --- Executes on button press in pastebutton.
function pastebutton_Callback(hObject, eventdata, handles)
% hObject    handle to pastebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if ~isempty(handles.clipboard)
    if ~iscell(handles.clipboard)   %length(handles.clipboard) == 1 %#ok<ISMT>
        handles = deselectprevious(handles);
        handles.selected = impoly(handles.hFigAxis, handles.clipboard);
        disp('Pasted annotation from the clipboard.');
    else
        handles.polys = [handles.polys handles.clipboard];
        if ~isempty(handles.hline)
            for i = 1:length(handles.hline)
                delete(handles.hline{i});
            end
        end
        handles.hline = [ ];
        axes(handles.hFigAxis); %#ok<MAXES>
        hold on; 
        for p = 1:length(handles.polys)
            pol = handles.polys{p};
            handles.hline{p} = plot([pol(:,1); pol(1,1)],[pol(:,2); pol(1,2)],'r','LineWidth',2);
        end
        hold off;
        disp('Pasted MULTIPLE ANNOTATIONS from the clipboard.');
    end
end
guidata(hObject, handles);


% ================ SAVE ANNOTATIONS TO XML ================================
% --- Executes on button press in savebutton.
function savebutton_Callback(hObject, eventdata, handles)
% hObject    handle to savebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if ~isempty(handles.polys)
    
    [pathstr, name1, ext, versn] = fileparts(handles.Ifilename); %#ok<NASGU>
    [pathstr, name2, ext, versn] = fileparts(handles.XMLfilename); %#ok<NASGU>
    
    if strcmp(name1, name2)

        polys2xml(handles.polys, handles.Ifilename, handles.XMLfilename, handles.imagefolder, handles.xmlfolder, 'mitochondria', handles.IMSIZE);

    else
        warning(['Image and XML filenames did not match -- did not save annotation to ' handles.XMLfilename]); %#ok<WNTAG>
    end
end












function xmlfilter_Callback(hObject, eventdata, handles)
% hObject    handle to xmlfilter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function namefiltertxt_Callback(hObject, eventdata, handles) %#ok<DEFNU>
% hObject    handle to namefiltertxt (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

function namefiltertxt_CreateFcn(hObject, eventdata, handles)

function xmlfilter_CreateFcn(hObject, eventdata, handles)

% --- Executes on button press in debughandles.
function debughandles_Callback(hObject, eventdata, handles) %#ok<DEFNU>
% hObject    handle to debughandles (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


handles %#ok<NOPRT>
keyboard;





% ======================= Z SLIDER ======================================
% --- Executes on slider movement.
function zslider_Callback(hObject, eventdata, handles)
% hObject    handle to zslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
Zval = round(get(hObject, 'Value'));
Zmin = get(hObject,'Min');
set(handles.z, 'String', num2str(Zval));

% save your current XML annotations (if you have any)
if ~isempty(handles.polys)
    %disp('Saving annotations!');
    savebutton_Callback(handles.savebutton, [ ], handles);
else
    disp('No annotations to save!');
end

% load the next image
file_ind = Zval - Zmin + 1;
filenm = handles.d_image(file_ind).name;
fullfile = [handles.imagefolder filenm];
set(handles.hFig, 'Name', filenm);
handles.I = imread(fullfile);
handles.api.replaceImage(handles.I, 'PreserveView', 1);
handles.Ifilename = filenm;

% delete the current annotations
if ~isempty(handles.hline)
    for i = 1:length(handles.hline);
        delete(handles.hline{i});
    end
end
handles.hline = [];
handles.polys = [];

% check and see if there is a corresponding XML annotation
handles = checkIMloadXML(handles);
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function zslider_CreateFcn(hObject, eventdata, handles)
% hObject    handle to zslider (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end



% =================== TRY TO LOAD AN XML FILE IF IT EXISTS ===============
function handles = checkIMloadXML(handles)

[pathstr, name, ext, versn] = fileparts(handles.Ifilename); %#ok<NASGU>
xmlfile = [name '.xml'];
xmlfullfile = [handles.xmlfolder xmlfile];

% if the file exists, load the annotations and darw them
if exist(xmlfullfile, 'file') 
    
    disp(['reading ' xmlfullfile]);
    [handles, polys] = getpolysfromxml(handles, xmlfullfile);
    if handles.show == 1
        axes(handles.hFigAxis); %#ok<MAXES>
        hold on; 
        for p = 1:length(polys)
            pol = polys{p};
            handles.hline{p} = plot([pol(:,1); pol(1,1)],[pol(:,2); pol(1,2)],'r','LineWidth',2);
        end
        hold off;
    end
    handles.polys = polys; 
    handles.XMLfilename = xmlfile;
else
    disp(['XML file DOES NOT EXIST: ' xmlfile]);
    handles.XMLfilename = xmlfile;
end


function [handles, polys] = getpolysfromxml(handles, filename)
% load annotation file:
v = loadXML(filename);
%disp(['reading ' filename]);
polys = {};

Nobjects = length(v.annotation.object); n=0;
for i = 1:Nobjects
    if v.annotation.object(i).deleted == '0'
        n = n+1;
        X = str2num(char({v.annotation.object(i).polygon.pt.x})); %#ok<*ST2NM> %get X polygon coordinates
        Y = str2num(char({v.annotation.object(i).polygon.pt.y})); %get Y polygon coordinates
        polys{n} = [X Y]; %#ok<AGROW>
    end
end






%==========================================================================
% --- Executes on a key press on hFig - the figure containing the image
%==========================================================================
% --- Executes on key press with focus on control_figure and no controls selected.
function keypresshandler(src, evnt, handles)

src = handles.Annotator3D;
handles = guidata(src);

disp(['=>you pressed "' evnt.Key '"']);
switch evnt.Key
    case 'uparrow'
        zvalue = min(get(handles.zslider, 'Max'), get(handles.zslider, 'Value') + 1);
        set(handles.zslider, 'Value', zvalue);
        zslider_Callback(handles.zslider, [ ], handles);
    case 'downarrow'
        zvalue = max(get(handles.zslider, 'Min'), get(handles.zslider, 'Value') - 1);
        set(handles.zslider, 'Value', zvalue);
        zslider_Callback(handles.zslider, [ ], handles);
    case 'delete'
        deletepoly_Callback(handles.deletepoly, [ ], handles);
    case 'return'
        newpolybutton_Callback(handles.newpolybutton, [], handles);
    case 'c'
        copybutton_Callback(handles.copybutton, [], handles);
    case 'v'
        pastebutton_Callback(handles.pastebutton, [], handles);
    case 'space'
        if isempty(handles.selected)
            selectbutton_Callback(handles.selectbutton, [], handles);
        else
            unselectbutton_Callback(handles.unselectbutton, [], handles);
        end
    case 'shift'
        showhidebutton_Callback(handles.showhidebutton, [], handles);
    case 's'
        savebutton_Callback(handles.savebutton, [], handles);
    case 'h'
        helpbutton_Callback(handles.helpbutton, [ ], handles);
    otherwise
        disp('unrecognized key command');
end
%guidata(src, handles);


% --- Executes on button press in helpbutton.
function helpbutton_Callback(hObject, eventdata, handles)
% hObject    handle to helpbutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

disp(' ');
disp('------keyboard shortcut help-------')
disp('up      up arrow increases the Z slice');
disp('down    down arrow decreases the Z slice');
disp('space   selects/unselects an annotation');
disp('enter   adds a new annotation');
disp('delete  deletes a selected annotation');
disp('shift   shows/hides the annotations');
disp('c       copies a selected annotation');
disp('v       pastes annotation(s)');
disp('s       saves annotations to an XML file');
disp('h       shows help information');
disp('-----------------------------------')
disp(' ');



% --- Executes on button press in showhidebutton.
function showhidebutton_Callback(hObject, eventdata, handles)
% hObject    handle to showhidebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if handles.show == 1
    % hide all the annotations
    
    % delete the current annotations
    if ~isempty(handles.hline)
        for i = 1:length(handles.hline);
            delete(handles.hline{i});
        end
    end
    handles.hline = [];
    disp('Hiding the annotations.');
%     if ~isempty(handles.selected)
%         unselectbutton_Callback(handles.unselectbutton, [], handles)
%     end
    handles.show = 0;
else
    % show all the annotations
    polys = handles.polys;
    axes(handles.hFigAxis); %#ok<MAXES>
    hold on; 
    for p = 1:length(polys)
        pol = polys{p};
        handles.hline{p} = plot([pol(:,1); pol(1,1)],[pol(:,2); pol(1,2)],'r','LineWidth',2);
    end
    hold off;
    disp('Showing the annotations.');
    handles.show = 1;
end
guidata(hObject, handles);


% --- Executes on button press in interpolatebutton.
function interpolatebutton_Callback(hObject, eventdata, handles)
% hObject    handle to interpolatebutton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% check if annotation files above exist
Zval = round(get(handles.zslider, 'Value'));
Zmax = get(handles.zslider,'Max');
Zmin = get(handles.zslider,'Min');
Ztarget = Zval + 1;
if Zval < Zmax
    file_ind = Zval - Zmin + 2;
    filenm = handles.d_image(file_ind).name;
    
    [pathstr, name, ext, versn] = fileparts(filenm); %#ok<NASGU>
    xmlupfile = [handles.xmlfolder name '.xml'];
    if exist(xmlupfile, 'file')
        [handles, polysA] = getpolysfromxml(handles, xmlupfile);
        disp(['reading ' xmlupfile]);
    else
        disp(['No annotations found for '  num2str(Ztarget) ]);
        return;
    end
else
   disp(['No annotations found for '  num2str(Ztarget) ]);
   return;
end

% check if annotation files below exist
Ztarget = Zval - 1;
if Zval > Zmin
    file_ind = Zval - Zmin;
    filenm = handles.d_image(file_ind).name;
    
    [pathstr, name, ext, versn] = fileparts(filenm); %#ok<NASGU>
    xmldnfile = [handles.xmlfolder name '.xml'];
    if exist(xmldnfile, 'file')
        [handles, polysB] = getpolysfromxml(handles, xmldnfile);
        disp(['reading ' xmldnfile]);
    else
        disp(['No annotations found for '  num2str(Ztarget) ]);
        return;
    end
else
   disp(['No annotations found for '  num2str(Ztarget) ]);
   return;
end


% check to see if we have any annotations on the current slice
if ~isempty(handles.polys)
    r = input(['\nChoose an action: (1) Delete or (2) keep current annotations for slice ' num2str(Zval) '?\n']);
    if r==2
        disp('You have decided to keep the annotations!');
    else
        disp('Annotations deleted!');
        handles.polys = [];    
        if ~isempty(handles.hline)
            for i = 1:length(handles.hline);
                delete(handles.hline{i});
            end
        end
        handles.hline = [];
    end
end

INTERP_POLYS = {};

% loop through A polys
for a = 1:length(polysA)
    % select interior points
    pol = polysA{a};
    NPOINTS = 15;
    
    p1 = randsample(1:size(pol,1), NPOINTS, 1);
    p2 = randsample(1:size(pol,1), NPOINTS, 1);
    
    pts = (pol(p1,:) + pol(p2,:) )/2;
%     axes(handles.hFigAxis); %#ok<MAXES>
%     hold on; 
%     plot(pts(:,1),pts(:,2),'r*');
%     hold off;
    
    % find best matching poly from B
    blist = zeros(size(polysB));
    for b = 1:length(polysB)
        x = pts(:,1); y = pts(:,2);
        blist(b) = sum(inpoly([x y], polysB{b}));
        %blist(b) = sum(in);
    end
    bmatch = find(blist == max(blist));
    polyb = polysB{bmatch}; %#ok<FNDSB>
    bw = poly2mask(polyb(:,1),polyb(:,2),handles.IMSIZE(1),handles.IMSIZE(2));
    bw = bwperim(bw);
    [Br Bc] = find(bw);
    
    %keyboard;
    
    % sample points from A and interpolate to B
    pol = [pol; pol(1,:)];
    DIST = 8;
    poly_interp = [];
    for p = 1:size(pol,1)-1  % loop through vertices in the poly of A
        
        d = euclidean(pol(p,:), pol(p+1,:));
        
        alist = 0:max(0, round(d/DIST));
        alist = alist/(length(alist));
        a1 = pol(p,:);
        a2 = pol(p+1,:);
        alist = ((a2-a1)'*alist)' + (a1'*ones(size(alist)))';
        
        for apt = 1:size(alist,1)
            ca = alist(apt,1);
            ra = alist(apt,2);
            dists = sqrt((Br - repmat(ra,size(Br,1), size(Br,2))).^2 + (Bc - repmat(ca, size(Bc,1), size(Bc,2))).^2);
            ind = find(dists == min(dists),1, 'first');
                
            %int = [ca ra];
            %int = [Bc(ind) Br(ind)];
            %int = [((ra + Br(ind))/2) ((ca + Bc(ind))/2)];
            int = [round((ca + Bc(ind))/2) round((ra + Br(ind))/2)];
            poly_interp = [poly_interp; int]; %#ok<AGROW>
            %keyboard;
        end
        %keyboard;
    end
    INTERP_POLYS{a} = poly_interp; %#ok<AGROW>
end

% add the polys to the master list
m = 1;
for n = length(handles.polys)+1 : length(handles.polys)+length(INTERP_POLYS)
    handles.polys{n} = INTERP_POLYS{m}; m = m + 1;
end



% delete all shown annotations
if ~isempty(handles.hline)
    for i = 1:length(handles.hline);
        delete(handles.hline{i});
    end
end
handles.hline = [];

% now draw all the annotations
polys = handles.polys;
axes(handles.hFigAxis); %#ok<MAXES>
hold on; 
for p = 1:length(polys)
    pol = polys{p};
    handles.hline{p} = plot([pol(:,1); pol(1,1)],[pol(:,2); pol(1,2)],'r','LineWidth',2);
end
hold off;

handles.show = 1;
guidata(hObject, handles);




function d = euclidean(p1, p2)

d = sqrt( (p1(1) - p2(1))^2 + (p1(2) - p2(2))^2); 
