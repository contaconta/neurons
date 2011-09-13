function [RUN, happyVector] = HappyNeuronVector(RUN)


HAPPY_THRESHOLD = 50; %200; %25;  %100;

% By default neurons are unhappy
happyVector = zeros(size(RUN.tracks));

%% We only care for cells that have a given number of filaments
for nTrack = 1:length(RUN.trkSeq)
   if isempty(RUN.trkSeq{nTrack})
       continue
   end
   track = RUN.trkSeq{nTrack};
   mean_neurite_t = zeros(size(track));
   mean_number_d = zeros(size(track));
   soma_perimeter_t = zeros(size(track));

   for nDtc = 1:length(track)
       d = track(nDtc);
       nmbr_dendrites = max(RUN.FILAMENTS(d).NeuriteID);
       dendrite_lengths = zeros(1,nmbr_dendrites);
       for nddr = 1:nmbr_dendrites
           dendrite_lengths(nddr) = size(find(RUN.FILAMENTS(d).NeuriteID == nddr),1);
       end
       if(isempty(nmbr_dendrites) || nmbr_dendrites ==0 )
         mean_number_d(d) = 0;
         mean_neurite_t(d) = 0;       
       else
        mean_number_d(d) = nmbr_dendrites;        
        mean_neurite_t(d) = mean(dendrite_lengths);
       end
       soma_perimeter_t(d) = RUN.Soma(d).Perimeter;
   end
   % if they are not many, the are long and variate, they have fun
   %happy_factor = mean(mean_neurite_t)*std(mean_neurite_t)/(mean(mean_number_d)+1);
   happy_factor = mean(mean_neurite_t)*std(mean_neurite_t)*std(soma_perimeter_t);
   happy_factor = (mean(mean_neurite_t)*std(mean_neurite_t)*std(soma_perimeter_t))/(mean(mean_number_d)+1);;
   
%    happy_factor_list(nTrack) =  happy_factor;
   
   if(happy_factor > HAPPY_THRESHOLD)
      happyVector(track) = 1;
%        str = sprintf('Neuron %i HAPPY %f\n', D(track(1)).ID, happy_factor);
%        disp(str);
   else
       
       
       
%        str = sprintf('Neuron %i SAD %f\n', D(track(1)).ID, happy_factor);
%        disp(str);
   end
   
end

for i = 1:length(RUN.D)
   RUN.D(i).Happy = happyVector(i); 
end

% low_list = happy_factor_list < 1000;
% 
% find(low_list)
% happy_factor_list(low_list)
%keyboard;
