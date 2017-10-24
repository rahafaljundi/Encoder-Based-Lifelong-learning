function net=get_orgin_from_freezed_network(old_net_path,freezed_net_path)
% Recover orginial network from a freezed one (see Model_preparation
% for mode details)
%
% Author: Rahaf Aljundi
%
% See the COPYING file.

old_net=load(old_net_path) ;
if(isfield(old_net,'net'))
    old_net=old_net.net;
end
lwf_net=load(freezed_net_path) ;
if(isfield(lwf_net,'net'))
    lwf_net=lwf_net.net;
end
%till the custom layer
net.layers=lwf_net.layers(1:end-2);
net.layers(end+1:end+2)=old_net.layers(end-1:end);
net.meta=old_net.meta;
end