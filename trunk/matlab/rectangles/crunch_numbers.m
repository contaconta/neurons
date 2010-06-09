function crunch_numbers(N, folder)


fplocs = [1.1e-6:.1e-6:1e-5,1.1e-5:.1e-5:1e-4,1.1e-4:.1e-4:1e-3,1.1e-3:.1e-3:1e-2, 1.1e-2:.1e-2:1e-1, 1.1e-1:.1e-1:1];
matfilename = ['TP' num2str(N) '.mat'];

% check to see if results already exist for this N
if exist(matfilename, 'file')
    load(matfilename);
else
    prefix_list = {};
    file_list = {};
    TP_list = zeros(0,length(fplocs));
end

% scan the files in the directory
d = dir([folder '*.mat']);
for i = 1:length(d)
    filename = d(i).name;
    if ~ismember(filename, file_list);   % if it is not a member we should process it
        
        % get the files prefix, and try to analyze it
        prefix = extract_prefix(filename);
        if ~isempty(prefix)
            [TP_list, file_list, prefix_list] = analyze_file(folder, filename, N, fplocs, TP_list, prefix, prefix_list, file_list, matfilename);
        else
            disp(['skipped ' filename]);
        end
        
    end
end

available_prefix = unique(prefix_list);




%% plottting!
figure; hold on;
for i = 1:length(available_prefix)
    [tf, members] = ismember(prefix_list, available_prefix(i));
    inds = find(members);
    files = file_list(inds);
    TP = TP_list(inds,:);    
    plot(fplocs, mean(TP,1), 'b-');
end

keyboard;




function prefix = extract_prefix(filename)


prefix = 'VJ-'; if strfind(filename, prefix); return; end;
prefix = 'A2-'; if strfind(filename, prefix); return; end;
prefix = 'A4-'; if strfind(filename, prefix); return; end;
prefix = 'A8-'; if strfind(filename, prefix); return; end;
prefix = 'A12-'; if strfind(filename, prefix); return; end;
prefix = '50-50'; if strfind(filename, prefix); return; end; 
prefix = '33-'; if strfind(filename, prefix); return; end;
prefix = 'VJANORM-'; if strfind(filename, prefix); return; end;
prefix = 'VJDNORM-'; if strfind(filename, prefix); return; end;
prefix = 'VJSPECIAL-'; if strfind(filename, prefix); return; end;
prefix = 'Simple2-'; if strfind(filename, prefix); return; end;
prefix = 'Simple4-'; if strfind(filename, prefix); return; end;
prefix = 'Simple8-'; if strfind(filename, prefix); return; end;
prefix = 'Simple12-'; if strfind(filename, prefix); return; end;
prefix = 'Amix-'; if strfind(filename, prefix); return; end;
prefix = 'Amix25-'; if strfind(filename, prefix); return; end;
prefix = 'Amix33-'; if strfind(filename, prefix); return; end;
prefix = 'Amix50-'; if strfind(filename, prefix); return; end;
prefix = 'Amix25Disconnect-'; if strfind(filename, prefix); return; end;
prefix = 'K2-'; if strfind(filename, prefix); return; end;
prefix = 'K4-'; if strfind(filename, prefix); return; end;
prefix = 'K8-'; if strfind(filename, prefix); return; end;
prefix = 'K12-'; if strfind(filename, prefix); return; end;
prefix = 'RANKFIX2-'; if strfind(filename, prefix); return; end;
prefix = 'RANKFIX4-'; if strfind(filename, prefix); return; end;
prefix = 'RANKFIX8-'; if strfind(filename, prefix); return; end;
prefix = 'RANKFIX12-'; if strfind(filename, prefix); return; end;
%prefix = 'LIENHART_NONORM-'; if strfind(filename, prefix); return; end;
%prefix = 'LIENHART_ANORM-'; if strfind(filename, prefix); return; end;

prefix = []; % if none are chose, return empty

function  [TP_list, file_list, prefix_list] = analyze_file(folder, filename, N, fplocs, TP_list, prefix, prefix_list, file_list, matfilename)

disp(['analyzing ' filename]);
load([folder filename]);
if length(CLASSIFIER.rects) >= N    % we should only process if it has sufficient learners
    [TP FP NP NN] = evaluate_test_set(CLASSIFIER, N, filename);
    TP_list(length(TP_list)+1,:) = interp1(FP/NN,TP/NP,fplocs); %#ok<AGROW>
    %TP_list(size(TP_list,1)+1,:) = rand(1,size(TP_list,2)); 
    file_list{size(TP_list,1)} = filename;  %#ok<AGROW>
    prefix_list{size(TP_list,1)} = prefix;
    
    disp(['saving ' matfilename]);
    save(matfilename);
end

