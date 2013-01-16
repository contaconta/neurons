function B = SigmoidFitting(PositiveEMDs, NegativeEMDs)

listOfNegToUse = randi(numel(NegativeEMDs), numel(PositiveEMDs), 1);
X = [PositiveEMDs; NegativeEMDs(listOfNegToUse)];
Y = zeros(size(X));
Y(1:length(X)/2) = 1;
B = glmfit(X, [Y ones(size(Y))], 'binomial', 'link', 'logit');
