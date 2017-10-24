function [net, info] = cnn_fine_tune_freeze(varargin)
%CNN_FINE_TUNE_FREEZE finetune the last layers of the network for
%the new task before adding it to the global model
%See add_new_task for details
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
    '..', 'matlab', 'vl_setupnn.m')) ;
opts.train.gpus=[1];
opts.train.learningRate=0.01*ones(1,100);
opts.train.batchSize=256;
opts.dataDir = fullfile(vl_rootnn, 'data','ILSVRC2012') ;
opts.modelType = 'alexnet' ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
opts.imdbPath = fullfile(opts.dataDir,'imdb');
opts.batchSize=256;
opts.modelPath='data/models/imagenet-caffe-alex.mat';
opts.compute_stats=false;
opts.freeze_layer=15;
opts.add_dropout=true;
opts.useGpu=false;
opts.weightInitMethod = 'gaussian';
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.useValidation=true;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.dataDir, opts.imdbPath);
opts.numAugments=3;



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
%==========================================================================
if(opts.compute_stats)
% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
    train = find(imdb.images.set == 1) ;
    images = fullfile( imdb.images.data(train(1:100:end))) ;
    [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
        'imageSize', [256 256], ...
        'numThreads', opts.numFetchThreads, ...
        'gpus', opts.train.gpus) ;
    save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;
end
% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = load(opts.modelPath );
if(isfield(net,'net'))
    net=net.net;
end


% Meta parameters
if(opts.compute_stats)

net.meta.normalization.averageImage=averageImage;
end
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;

net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * zeros(3)) ;
net.meta.augmentation.jitterAspect = [2/3, 3/2] ;

net.meta.trainOpts.learningRate =opts.train.learningRate ;
net.meta.trainOpts.numEpochs = numel(opts.train.learningRate) ;
net.meta.trainOpts.batchSize = opts.train.batchSize ;
net.meta.trainOpts.weightDecay = 0.0005 ;
%---------------------------------------------------------------------------
%                                                           put drop out layers
%---------------------------------------------------------------------------
aux_net=net;
opts.scale = 1 ;
if (opts.add_dropout)
aux_net.layers = net.layers(1:end-4);
%add dropout
aux_net = add_dropout(aux_net,  'drop_out6');
%get number of classes from the dataset

%move fc7 and relu
aux_net.layers{end+1}= net.layers{end-3};
aux_net.layers{end+1}= net.layers{end-2};
%add dropout
aux_net = add_dropout(aux_net,  'drop_out7');

else
   aux_net.layers = net.layers(1:end-2); 
end
%add task specific layer
aux_net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{init_weight(opts,1,1,4096,numel(imdb.meta.classes), 'single'), zeros(1,numel(imdb.meta.classes),'single')}}, ...
    'learningRate', [1 1], ...
    'stride', [1 1], ...
    'pad', [0 0 0 0]) ;
aux_net.layers{end+1} = struct('type', 'softmaxloss') ;%

%FREEZE OLD LAYERS
for i=1:opts.freeze_layer-1
    
    
    if strcmp(aux_net.layers{1,i}.type,'conv')
     
        aux_net.layers{1,i}.learningRate=[0 0];
        
    end
end

net=aux_net;
net = vl_simplenn_tidy(net) ;


% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainFn = @cnn_train ;
    case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat');

switch opts.networkType
    case 'simplenn'
        save(modelPath, '-struct', 'net') ;
    case 'dagnn'
        net_ = net.saveobj() ;
        save(modelPath, '-struct', 'net_') ;
        clear net_ ;
end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.
switch lower(opts.weightInitMethod)
    case 'gaussian'
        sc = 0.01/opts.scale ;
        weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
        sc = sqrt(3/(h*w*in)) ;
        weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
        sc = sqrt(2/(h*w*out)) ;
        weights = randn(h, w, in, out, type)*sc ;
    otherwise
        error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end


function net = add_dropout(net,  id)
% --------------------------------------------------------------------

net.layers{end+1} = struct('type', 'dropout', ...
    'name', sprintf('dropout%s', id), ...
    'rate', 0.5) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
    mu = double(meta.normalization.averageImage(:)) ;
else
    mu = imresize(single(meta.normalization.averageImage), ...
        meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
    'useGpu', useGpu, ...
    'numThreads', opts.numFetchThreads, ...
    'imageSize',  meta.normalization.imageSize(1:2), ...
    'cropSize', meta.normalization.cropSize, ...
    'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
    f = char(f) ;
    bopts.train.(f) = meta.augmentation.(f) ;
end
bopts.numAugments=opts.numAugments;
fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(batch) ;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
    phase = 'train' ;
else
    phase = 'test' ;
end
%handelling the augmentation
data=[];
for i=1:opts.numAugments
    curr_data = getImageBatch(images, opts.(phase), 'prefetch', nargout == 0) ;
    if(~isempty(data))
        data=cat(4,data,curr_data);
    else
        data=curr_data;
    end
    
end
rep_inds=get_aug_inds(opts,batch);
data=data(:,:,:,rep_inds);

if nargout > 0
    labels = imdb.images.labels(batch) ;
    
    a=repmat(labels,size(data,4)/size(batch,2), 1);
    labels=a(:);

    switch networkType
        case 'simplenn'
            varargout = {data, labels} ;
        case 'dagnn'
            varargout{1} = {'input', data, 'label', labels} ;
    end
end




