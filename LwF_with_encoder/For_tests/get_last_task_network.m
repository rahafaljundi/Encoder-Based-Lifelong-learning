function net=get_last_task_network(lwf_net_path)
% loads the network of the latest trained task
% Author: Rahaf Aljundi
%
% See the COPYING file.

lwf_net=load(lwf_net_path) ;
if(isfield(lwf_net,'net'))
    lwf_net=lwf_net.net;
end

%till the custom layer
index=2;
for i=1:numel(lwf_net.layers)
    if(strfind(lwf_net.layers{i}.type,'custom'))
       index =i-1;
       break
    end
end
net.layers=lwf_net.layers(1:index);
net.layers(end+1:end+numel(lwf_net.layers{end}.tasks{end}.layers))=lwf_net.layers{index+1}.tasks{end}.layers(1:end);
net.meta=lwf_net.meta;
end