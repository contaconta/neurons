function trkWriteXMLFile(S, xmlFileName)




docNode = com.mathworks.xml.XMLUtils.createDocument('Root');

docRootNode = docNode.getDocumentElement;
docRootNode.appendChild(docNode.createComment(['Neuron morphodynamic analysis. Performed ' date '. Copyright EPFL. All rights reserved.']));


 %% Root elements
 
    
    names = fieldnames(S);

    for n = 1:length(names)
        if ~strcmpi(names{n}, 'TimeStep')
            val = S.(names{n});
            node = docNode.createElement(names{n});
            node.appendChild(docNode.createTextNode(num2str(val)));
            docRootNode.appendChild(node);
        end
    end


% % the experiment label
% Label = docNode.createElement('Label');
% Label.appendChild(docNode.createTextNode(S.Label));
% docRootNode.appendChild(Label);
% 
% % the number of cells
% node = docNode.createElement('CellsTracked');
% node.appendChild(docNode.createTextNode(num2str(S.NumberOfCells)));
% docRootNode.appendChild(node);
% 
% % the sequnce length
% node = docNode.createElement('SequenceLength');
% node.appendChild(docNode.createTextNode(num2str(S.Length)));
% docRootNode.appendChild(node);
% 
% % the Date
% node = docNode.createElement('ExperimentDate');
% node.appendChild(docNode.createTextNode(S.Date));
% docRootNode.appendChild(node);

for t=1:S.Length
    TimeNode = docNode.createElement('Image');
    TimeNode.setAttribute('Time',sprintf('%i', t));
    %TimeNode.appendChild(docNode.createTextNode(sprintf('%i', t)));
    
    %disp(['writing t = ' num2str(t) ' of ' xmlFileName]);
    
    
    
    %% ImageT elements
    ImageT = S.TimeStep(t);
    
    names = fieldnames(ImageT);

    for n = 1:length(names)
        if ~strcmpi(names{n}, 'Neuron') && ~strcmpi(names{n}, 'Time')
            val = ImageT.(names{n});
            node = docNode.createElement(names{n});
            node.appendChild(docNode.createTextNode(num2str(val)));
            TimeNode.appendChild(node);
        end
    end

    
    
    Neurons = S.TimeStep(t).Neuron;
    for i = 1:length(Neurons)
        neuron = docNode.createElement('Neuron');
        neuron.setAttribute('ID', num2str(Neurons(i).Nucleus.ID));
        
        % create the soma and nucleus
        Soma = S.TimeStep(t).Neuron(i).Soma;
        Nucleus = S.TimeStep(t).Neuron(i).Nucleus;        
        snode = docNode.createElement('Soma');
        nnode = docNode.createElement('Nucleus');
        
        
        %% nucleus elements
        names = fieldnames(Nucleus);
        
        for n = 1:length(names)
            if ~strcmpi(names{n}, 'Centroid') && ~strcmpi(names{n}, 'ID')
                val = Nucleus.(names{n});
                node = docNode.createElement(names{n});
                node.appendChild(docNode.createTextNode(num2str(val)));
                nnode.appendChild(node);
            end
        end
        val = Nucleus.Centroid(1);
        node = docNode.createElement('X');
        node.appendChild(docNode.createTextNode(num2str(val)));
        nnode.appendChild(node);
        val = Nucleus.Centroid(2);
        node = docNode.createElement('Y');
        node.appendChild(docNode.createTextNode(num2str(val)));
        nnode.appendChild(node);
        
        %% soma elements
        names = fieldnames(Soma);
        
        for n = 1:length(names)
            if ~strcmpi(names{n}, 'Centroid') && ~strcmpi(names{n}, 'ID')
                val = Soma.(names{n});
                node = docNode.createElement(names{n});
                node.appendChild(docNode.createTextNode(num2str(val)));
                snode.appendChild(node);
            end
        end
        val = Soma.Centroid(1);
        node = docNode.createElement('X');
        node.appendChild(docNode.createTextNode(num2str(val)));
        snode.appendChild(node);
        val = Soma.Centroid(2);
        node = docNode.createElement('Y');
        node.appendChild(docNode.createTextNode(num2str(val)));
        snode.appendChild(node);
        
        
        % attach the soma and nucleus
        neuron.appendChild(nnode);
        neuron.appendChild(snode);
        
        % attach the neuron
        TimeNode.appendChild(neuron);
    end

    % attach the timestep
    docRootNode.appendChild(TimeNode);

end


%docRootNode.setAttribute('folder',folder);

%thisElement = docNode.createElement('child_node');
%docRootNode.appendChild(thisElement);
%docNode.appendChild(docNode.createComment(['Sinergia neuron morphodynamic analysis. Performed ' date '. Copyright EPFL. All rights reserved.']));


 % Save the sample XML document.
%xmlFileName = [folder 'track_data.xml'];
xmlwrite(xmlFileName,docNode);


