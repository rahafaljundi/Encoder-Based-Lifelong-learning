function [net, info] = cnn_lwf_with_encoder(varargin)
%CNN_LWF_WITH_ENCODER  Demonstrates training a CNN with the Encoder Based
%Lifelong Learning method after preparing the model(See functions under
%Model_prepapration). 
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

run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..','matlab', 'vl_setupnn.m')) ;
opts.train.gpus=[1];
opts.train.learningRate=1e-4*ones(1,100);
opts.train.batchSize=100;
opts.expDir = fullfile(vl_rootnn, 'data', 'LwF_with_encoder') ;
opts.dataDir = fullfile(vl_rootnn, 'data', 'dataset-2') ;
opts.modelType = 'alexnet' ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
opts.imdbPath = fullfile(opts.expDir, 'aug-imdb-2.mat');
opts.modelPath=fullfile(opts.expDir, 'model-task-2-initial.mat');
opts.temperature=2;
[opts, varargin] = vl_argparse(opts, varargin) ;
sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.useValidation=false;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

imdb = load(opts.imdbPath) ;
if(isfield(imdb,'imdb'))
    imdb=imdb.imdb;
end
mkdir(opts.expDir);
%--------------------------------------------------------------------------
%                                                   create a validation set
%--------------------------------------------------------------------------
if(opts.useValidation)
    sets=unique(imdb.images.set);
    if(numel(sets)==2)
        test_set=find(imdb.images.set~=1);
        imdb.images.set(test_set)=3;
        training_inds=find(imdb.images.set==1);
        training_size=numel(training_inds);
        %create validation inds
        val_inds= randi(training_size,floor(training_size/10),1);
        imdb.images.set(training_inds(val_inds))=2;      
    end
else
    test_set=find(imdb.images.set~=1);
    imdb.images.set(test_set)=2;
end
%The model already has the previous task information and parameters but
%make sure not to use the deployed model as a previous task model
% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
net = load(opts.modelPath );
if(isfield(net,'net'))
    net=net.net;
end
% Meta parameters
net.meta.trainOpts.learningRate =opts.train.learningRate ;
net.meta.trainOpts.numEpochs = numel(opts.train.learningRate) ;
net.meta.trainOpts.batchSize = opts.train.batchSize ;
net.meta.trainOpts.weightDecay = 0.0005 ;
% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainFn = @cnn_train_lwf_auto_fc_shared ;
    case 'dagnn', error('This function does not support DagNN yet. Please use simplenn');
end

[net, info] = trainFn(net, imdb, @getBatch, ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train) ;




% --------------------------------------------------------------------
function [img, labels,tasks_recs,tlabels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
%To extract batches from datasets that contain recorded codes and soft
%targets for previous tasks
%no augmentation is applied now, data augmented in preparation phase
ims = imdb.images.data(1,batch) ;
labels = imdb.images.labels(1,batch) ;
%here we have to add the codes and soft targets from each task
img=[];
%this is to be applied to already augmented data, that we store in an imdb
%file with image links
for i=1:numel(ims)
    load(ims{i});
    img= cat(4,img,im);
    clear im;
end
for t=1:numel(imdb.images.recs)
    if(~iscell(imdb.images.recs{t}))
        tasks_recs{t}=imdb.images.recs{t}(:,:,:,batch) ;
    else
        rec_links=imdb.images.recs{t}(batch) ;
        for i=1:numel(rec_links)
            load(rec_links{i});
            tasks_recs{t}= cat(4,tasks_recs{t},rec);
            clear rec;
        end
        
    end
end
for t=1:numel(imdb.images.tlabels)
    tlabels_all=imdb.images.tlabels{t};
    tlabels{t}=tlabels_all(batch,:)';
end
