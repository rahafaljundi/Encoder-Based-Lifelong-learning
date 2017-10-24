function net=get_orgin_from_lwf_network_fc_shared(task_number,lwf_net_path)
% loads the network of the trained task (task_number)
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
    if(findstr(lwf_net.layers{i}.type,'custom'))
       index =i;
       break
    end
end
net.layers=lwf_net.layers(1:index-1);
task_net=lwf_net.layers{index}.tasks{end};
index=2;
for i=1:numel(task_net.layers)
    if(findstr(task_net.layers{i}.type,'custom'))
       index =i;
       break
    end
end
net.layers(end+1:end+numel(task_net.layers(1:index-1)))=task_net.layers(1:index-1);
net.layers(end+1:end+numel(task_net.layers{index}.tasks{task_number}.layers(1:end-1)))=task_net.layers{index}.tasks{task_number}.layers(1:end-1);%fc_8
net.layers{end+1}=struct('type','softmax');
net.meta=lwf_net.meta;
end