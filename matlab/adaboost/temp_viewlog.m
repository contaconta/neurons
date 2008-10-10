function temp_viewlog(log_filenm)



DATA = logfile(log_filenm, 'read');


Di = DATA(:,6);
Fi = DATA(:,7);
di = DATA(:,8);
fi = DATA(:,9);
dit = DATA(:,10);
fit = DATA(:,11);
stages = DATA(:,1);


figure; hold on;



plot(Di, 'b-', 'LineWidth',2);
plot(Fi, 'r-', 'LineWidth',2);
plot(di, 'b-.');
plot(fi, 'r-.');
plot(dit, 'c-.');
plot(fit, 'm-.');
legend('Overall Detection Rate D_i', 'Overall False Positive Rate F_i', 'Stage Detection Rate d_i', 'Stage False Positive Rate f_i', 'Training Data d_i', 'Training Data f_i');
xlabel('# of Weak Learners');
ylabel('Detection Rate / False Positive Rate');
title('Cascade Learning Progress');
grid on;
ylim([-.1 1]);
xlim([1 size(DATA,1)]);

stagelist = unique(stages);
for i = stagelist'
    first = find(stages == i, 1, 'first');
    last = find(stages ==i, 1, 'last');
    fill([first last last first],[-.1 -.1 0 0], 2*[.3 .2 .5]);
    text(first+1, -.05,['stage ' num2str(i)]);
end


%fill([0 10 10 0],[-.1 -.1 0 0], [.3 .2 .5]);

%fill([0 x x 0],[-.1 -.1 0 0], [.3 .2 .5]);
%fill([0 1 1 0],[0 0 1 1], [.3 .2 .5])