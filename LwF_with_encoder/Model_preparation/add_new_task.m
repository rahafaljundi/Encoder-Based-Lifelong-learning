function [new_net, aug_imdb] = add_new_task(varargin)
%ADD_NEW_TASK modifies a model and  to prepare it for training  on a new task
%and modifies its corresponding data by:
%       -adding the two first layers of the autoencoder of the model
%       -augmenting the data if needed
%       -add the task specific layers to the model and train them for few
%       epochs for a better initialization.
%The last network layer is a custom layer, that contains 2 branches:
%       - 1 for the autoencoder
%       - 1 for the task operator (see paper for definition)
% The task operator is also divided into a commun part and a task specific
% part. The task specific part is the last layer, that is also a custom
% layer divided into a branch per task, in the same order the tasks are fed
% to the model.
%
%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko,
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Rahaf Aljundi
%
% See the COPYING file.
%
% Adapted from MatConvNet of VLFeat library. Their copyright info:
%
% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).


opts=getDefaultOpts();
[opts, ~] = vl_argparse(opts, varargin) ;

org_imdb=load(opts.imdb_path);

org_net=load(opts.orgin_net);
if(isfield(org_net,'net'))
    org_net=org_net.net;
end

%Prepare autoencoder
autoencoder_net = load(opts.autoencoder_path);
if isfield(autoencoder_net,'net')
    autoencoder_net=autoencoder_net.net;
end;
new_net=org_net;
new_enc_net.layers=[];
new_enc_net.layers{1}=struct('type','en_reshape');
new_enc_net.layers(end+1:end+2)=autoencoder_net.layers(1:2);
new_enc_net.layers{end+1}=struct('type','intermediate_euclideanloss');
new_enc_net.layers{end}.alpha=opts.alpha;
new_enc_net.layers{1}.gr_weight=1;

autoencoder_net=new_enc_net;
for i=1:numel(autoencoder_net.layers)
    if(isfield(autoencoder_net.layers{i},'Gf'))
        autoencoder_net.layers{i}=rmfield(autoencoder_net.layers{i},'Gf');
    end
    if(isfield(autoencoder_net.layers{i},'Df'))
        autoencoder_net.layers{i}=rmfield(autoencoder_net.layers{i},'Df');
    end
    autoencoder_net.layers{i}.learningRate=[0 0];
end

%Add autoencoder to the model
new_net.layers{end}.tasks{end+1}=new_net.layers{end}.tasks{end};
new_net.layers{end}.tasks{end-1}=autoencoder_net;

%Record the initial codes and last task targets for the new data
if(~exist(opts.output_imdb_path,'file'))
    mkdir(opts.aug_output_path)
    encoder_opts.output_layer_id=opts.split_layer;
    % if augmentation is needed
    if(opts.extract_images)
        org_imdb=load(opts.imdb_path);
        [aug_imdb]=augment_data(org_imdb,org_net,opts.aug_output_path,autoencoder_net,encoder_opts);
        [imdb] = record_task_aug(aug_imdb,new_net,opts);
        save(opts.output_imdb_path,'imdb');
        % if the data is already augmented
    else
        aug_imdb=load(opts.old_aug_imdb_path);
        if(isfield(aug_imdb,'imdb'))
            aug_imdb=aug_imdb.imdb;
        end
        [imdb] = record_task_aug(aug_imdb,new_net,opts);
        save(opts.output_imdb_path,'imdb');
    end
end


%finetune only the new task specific layer while freezing all other
fine_tune_opts.expDir=opts.freezeDir;
fine_tune_opts.imdbPath=opts.imdb_path;
fine_tune_opts.dataDir='';
fine_tune_opts.modelPath=opts.last_task_net_path;
fine_tune_opts.train.learningRate=opts.train.learningRate;
fine_tune_opts.train.batchSize=opts.train.batchSize;
fine_tune_opts.freeze_layer=22;
fine_tune_opts.add_dropout=0;
cnn_fine_tune_freeze(fine_tune_opts);
freezed_net=load(fullfile(opts.freezeDir,strcat('net-epoch-',num2str(findLastCheckpoint(opts.freezeDir)),'.mat')));
if isfield(freezed_net, 'net')
    freezed_net=freezed_net.net;
end;

%Add the knowledge distillation (old tasks) and cross entropy(new task)
%losses
new_net.layers{end}.tasks{end}.layers{end}.tasks{end}.layers{end}=struct('type','softmaxlossdiff');
new_net.layers{end}.tasks{end}.layers{end}.tasks{end+1}.layers{1}=freezed_net.layers{end-1:end};
new_net.layers{end}.tasks{end}.layers{end}.tasks{end}.layers{1}.dilate=1;
new_net.layers{end}.tasks{end}.layers{end}.tasks{end}.layers{end+1} = struct('type', 'softmaxloss') ;%

net=new_net;
save(opts.outputnet_path,'net');
end

% -------------------------------------------------------------------------
function opts=getDefaultOpts()
% -------------------------------------------------------------------------
%Initializes the options for the autoencoder training
%Make sure to change the paths to your data and models if needed

opts.split_layer=16;
opts.expDir = fullfile(vl_rootnn, 'data', 'LwF_with_encoder') ;
opts.orgNetPath= fullfile(opts.expDir, 'model-task-1.mat');
opts.imdb_path=fullfile(opts.expDir, 'imdb.mat');
opts.outputnet_path=fullfile(opts.expDir, 'model-task-2-initial.mat');
opts.output_imdb_path=fullfile(opts.expDir, 'imdb-2.mat');
opts.train.learningRate = [0.0004*ones(1, 54)  0.1*0.0004*ones(1, 18)] ;
opts.freezeDir=fullfile(opts.expDir, 'task-2-finetune');
opts.train.batchSize=15;
opts.aug_output_path=fullfile(vl_rootnn, 'data', 'aug-dataset-2') ;
opts.old_aug_imdb_path=fullfile(opts.expDir, 'aug-imdb-2.mat');
opts.autoencoder_dir = fullfile(vl_rootnn, 'data', 'autoencoder_with_classerror');
opts.autoencoder_path=fullfile(opts.autoencoder_dir, strcat('net-epoch-',num2str(findLastCheckpoint(opts.autoencoder_dir)),'.mat')) ;
opts.scale = 1 ;
opts.extract_images=true;
opts.alpha=1e-1;
opts.last_task_net_path='';%the last network after lwf
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
% Finds the network of the last epoch in the directory modelDir

list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end

