function temp_comparelogs(varargin)


for i = 1:nargin
    
    clear Di Fi di fi dit fit stages LEARNER;
    
    log_filenm = varargin{i};


    DATA = logfile(log_filenm, 'read');

    Di = DATA(:,4);
    Fi = DATA(:,5);
    di = DATA(:,6);
    fi = DATA(:,7);
    dit = DATA(:,8);
    fit = DATA(:,9);
    stages = DATA(:,1);
    LEARNER = DATA(:,10);


    %keyboard;
    
    if i == 1;
        figure; hold on;
    end


    plot(Di, 'b-', 'LineWidth',2);
    plot(Fi, color_select(i), 'LineWidth',2);
%     plot(di, 'b-');
%     plot(fi, 'r-');
%     plot(dit, 'c-');
%     plot(fit, 'm-');

    for l=1:max(LEARNER)
        str = learnerstr(l);
        plot(find(LEARNER==l), Di(LEARNER==l), str);
        plot(find(LEARNER==l), Fi(LEARNER==l), str);
    end

%     legend('Overall Detection Rate D_i', 'Overall False Positive Rate F_i', 'Stage Detection Rate d_i', 'Stage False Positive Rate f_i', 'Training Data d_i', 'Training Data f_i');
    xlabel('# of Weak Learners');
    ylabel('Detection Rate / False Positive Rate');
    title('Cascade Learning Progress');
    grid on;
    ylim([-.1 1]);
    %xlim([1 size(DATA,1)]);
    
    
    xlim([1 100]);

%     stagelist = unique(stages);
%     for i = stagelist'
%         first = find(stages == i, 1, 'first');
%         last = find(stages ==i, 1, 'last');
%         fill([first last last first],[-.1 -.1 0 0], 2*[.3 .2 .5]);
%         text(first+1, -.05,['stage ' num2str(i)]);
%     end

end


%% ================ SUPPORTING FUNCTIONS ================== 


function col = color_select(l)

switch l
    case 1
            col = 'r-';
    case 2
            col = 'b-';
    case 3
            col = 'g-';
    case 4
            col = 'k-';
    case 5
            col = 'm-';
    case 6
            col = 'c-';
    case 7
            col = 'y-';
    otherwise
            str = 'k:';
end

function str = learnerstr(l)

switch l
    case 1
            str = 'ko';
    case 2
            str = 'k*';
    case 3
            str = 'k.';
    case 4
            str = 'k+';
    case 5
            str = 'ks';
    case 6
            str = 'kd';
    case 7
            str = 'k^';
    case 8
            str = 'kx';
    case 9
            str = 'kp';
    otherwise
            str = 'kh';
end