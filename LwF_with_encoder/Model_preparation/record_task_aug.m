function [aug_imdb] = record_task_aug(imdb,net,opts)
%RECORD_TASK_AUG computed the codes and old tasks targets by applying net
%to tne new task data contained in imdb.
%net should contain: 
%   -old task autoencoders
%   -old task optimized task layers
%See add_new_task for more details.
%%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko, 
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Rahaf Aljundi & Amal Rannen Triki
% 
% See the COPYING file.

aug_imdb=imdb;
aug_imdb.images.labels =[];
aug_imdb.images.set =[];

for t=1:numel(net.layers{end}.tasks)-1
    net.layers{end}.tasks{t}.layers(end)=[];
    aug_imdb.images.recs{t}=[];
    
end
for t=1:numel(net.layers{end}.tasks{end}.layers{end}.tasks)
    net.layers{end}.tasks{end}.layers{end}.tasks{t}.layers{end}.type='softmax';
    aug_imdb.images.tlabels{t}=[];
    
end

inds = numel(imdb.images.data);
featues_net_opts.mode='test';
for counter=1:1000:inds
    aug_imdb.images.labels =[];
    aug_imdb.images.set =[];
    %To record the codes
    for t=1:numel(net.layers{end}.tasks)-1
        aug_imdb.images.recs{t}=[];
    end
    %To record the last tasks targets
    for t=1:numel(net.layers{end}.tasks{end}.layers{end}.tasks)    
        aug_imdb.images.tlabels{t}=[];        
    end
    last=counter+1000 -1;
    if(inds<last)
        last=inds;
    end
    for i = counter:last
        fprintf('Sample %d out of %d\n', i, inds);
        load(imdb.images.data{i});
        data=im;
        res=vl_simplenn_LwF_encoder(net,data,[],[],featues_net_opts);
        %============codes===================
        for t=1:numel(net.layers{end}.tasks)-1            
            this_res=res(end).aux{t}.layers{end}.x;                        
            scores =(this_res) ;            
            aug_imdb.images.recs{t}=cat(4,aug_imdb.images.recs{t},scores);            
        end
        %============codes===================
        %============targets=================
        for t=1:numel(net.layers{end}.tasks{end}.layers{end}.tasks)
            scores = squeeze(gather(res(end).aux{end}.layers{end}.aux{t}.layers{end}.x)) ;
            aug_imdb.images.tlabels{t}=[aug_imdb.images.tlabels{t};scores'];
        end
        %============targets=================        
        aug_imdb.images.labels=[ aug_imdb.images.labels imdb.images.labels(i)];
        aug_imdb.images.set=[ aug_imdb.images.set imdb.images.set(i)];        
    end
    save(strcat(num2str(counter),opts.output_imdb_path),'aug_imdb');
end
end
