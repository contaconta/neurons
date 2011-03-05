function [RUN, happyVector] = HappyNeuronVector(RUN)


HAPPY_THRESHOLD = 25;  %100;

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

   for nDtc = 1:length(track)
       nd = track(nDtc);
       nmbr_dendrites = max(RUN.FILAMENTS(nd).NeuriteID);
       dendrite_lengths = zeros(1,nmbr_dendrites);
       for nddr = 1:nmbr_dendrites
           dendrite_lengths(nddr) = size(find(RUN.FILAMENTS(nd).NeuriteID == nddr),1);
       end
       if(isempty(nmbr_dendrites) || nmbr_dendrites ==0 )
         mean_number_d(nDtc) = 0;
         mean_neurite_t(nDtc) = 0;       
       else
        mean_number_d(nDtc) = nmbr_dendrites;        
        mean_neurite_t(nDtc) = mean(dendrite_lengths);
       end
   end
   % if they are not many, the are long and variate, they have fun
   happy_factor = mean(mean_neurite_t)*std(mean_neurite_t)/(mean(mean_number_d)+1);
   
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

