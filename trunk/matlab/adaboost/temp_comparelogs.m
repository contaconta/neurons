function temp_comparelogs(varargin)


NEGCLASS = 10000;

for i = 1:nargin
    
    clear Di Fi di fi dit fit stages LEARNER;
    
    log_filenm = varargin{i};


    DATA = logfile(log_filenm, 'read');

    Di = DATA(:,4);
    %Fi = round(DATA(:,5)*NEGCLASS) -6;
    %Fi = round(DATA(:,5)*NEGCLASS);
    Fi = DATA(:,5);
    fi = DATA(:,7);
    dit = DATA(:,8);
    fit = DATA(:,9);
    stages = DATA(:,1);
    LEARNER = DATA(:,10);


    %keyboard;
    
    if i == 1;
        figure; hold on;
    end


%    plot(Di, 'b-', 'LineWidth',2);
    semilogy(Fi, color_select(i), 'LineWidth',1);
%     plot(di, 'b-');
%     plot(fi, 'r-');
%     plot(dit, 'c-');
%     plot(fit, 'm-');

    for l=1:max(LEARNER)
        str = learnerstr(l);
        %plot(find(LEARNER==l), Di(LEARNER==l), str);
        %semilogy(find(LEARNER==l), Fi(LEARNER==l), str);
    end

%     legend('Overall Detection Rate D_i', 'Overall False Positive Rate F_i', 'Stage Detection Rate d_i', 'Stage False Positive Rate f_i', 'Training Data d_i', 'Training Data f_i');
    xlabel('Number of Weak Learners');
    ylabel('False Positive Count');
    %title('Cascade Learning Progress');
    grid on;
    %ylim([0 .9]);
    %xlim([1 size(DATA,1)]);
    
     set(gca, 'YScale', 'log');
    
    xlim([1 200]);
    %axis([0 500 3 10000])
    
    legend('Haar', 'HoG', 'Spoke')
    
    %xlim([1 100]);

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
            col = 'g-';
    case 2
            col = 'm-';
    case 3
            col = 'b-';
    case 4
            col = 'k-';
    case 5
            col = 'c-';
    case 6
            col = 'r-';
    case 7
            col = 'y-';
    case 8  
            col = 'b-.';
    case 9
            col = 'k-.';
    otherwise
            col = 'k:';
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