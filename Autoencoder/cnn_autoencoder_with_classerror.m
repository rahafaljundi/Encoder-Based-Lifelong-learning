function [net, opts, imdb, info] = cnn_autoencoder_with_classerror(varargin)
%CNN_AUTOENCODER_WITH_CLASSERROR contructs an autoencoder to be trained in order to minimize:
%	- the reconstruction loss (as is classicly the case for autoencoders)
%	- the task loss (e.g. classification loss) when the task layers are given the output ofthe autoencoder
%For more details, see A. Rannen Triki, R. Aljundi, M. B. Blaschko, and T. Tuytelaars, Encoder Based Lifelong 
%Learning. ICCV 2017 - Section 3.
%
% Authors: Rahaf Aljundi & Amal Rannen Triki
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
opts = getAutoencoderOpts;
opts = vl_argparse(opts, varargin) ;

net  = get_onelayer_autoencoder(opts);
net = add_task_layers(net,opts);

if exist(opts.imdbPath, 'file')
    
    imdb=load(opts.imdbPath);
    if(isfield(imdb,'imdb'))
        imdb=imdb.imdb;
    end
    inds=find(imdb.images.set==2);
    if ~isempty(inds)
        opts.val = inds;
    end;
end

opts.compute_stats=false;
[net, info] = cnn_train_adadelta(net, imdb, getBatchFn(opts), opts);

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
end

% -------------------------------------------------------------------------
function net = get_onelayer_autoencoder(opts)
% -------------------------------------------------------------------------
%Constructs an autoencoder with one hidden layer, with sigmoid activation, and a reconstruction loss 
%multiplied by a parameter alpha that controls the compromise between the reconstruction loss and the 
%task loss

if (~isempty(opts.initial_encoder))
    load(opts.initial_encoder);
    net = vl_simplenn_move(net, 'cpu') ;
else
    net.layers = {} ;
    % Layer 1
    net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', 'code', '1'), ...
        'weights', {{init_weight(opts, 1, 1, opts.input_size,  opts.code_size, 'single'), ...
        ones( opts.code_size, 1, 'single')*opts.initBias}}, ...
        'stride', [1 1], ...
        'pad', [0 0 0 0], ...
        'dilate', 1, ...
        'learningRate', [1 1], ...
        'weightDecay', [1 0] ,'precious',1    ) ;
	%Sigmoid activation
    net.layers{end+1} = struct('name', 'encoder_1_sigmoid', ...
        'type', 'sigmoid'          );
    % Layer 2
    net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', 'data_hat', '3'), ...
        'weights', {{init_weight(opts, 1, 1, opts.code_size, opts.input_size, 'single'), ...
        ones(opts.input_size, 1, 'single')*opts.initBias}}, ...
        'stride', [1 1], ...
        'pad',  [0 0 0 0], ...
        'dilate', 1, ...
        'learningRate', [1 1], ...
        'weightDecay', [1 0]  , 'precious',1      ) ;
	%Loss
    net.layers{end+1} = struct('type', 'intermediate_euclideanloss','alpha',opts.alpha);
end

end

% -------------------------------------------------------------------------
function net = add_task_layers(net,opts)
% -------------------------------------------------------------------------
%Adds the task layers (e.g. fully connected layers for a classification ConvNet) to the autoencoder
%WARNING: orgNet.layers{end}.type should be set to the loss related to your task loss (e.g. softmaxlss).
%Make sure that this loss is interpretable by vl_simplenn.m

orgNet = load(opts.orgNetPath);
if(isfield(orgNet,'net'))
    orgNet=orgNet.net;
end
orgNet.layers{end}.type='softmaxloss';
orgNet = vl_simplenn_tidy(orgNet) ;
orgNet.layers(1:opts.output_layer_id-1) =[]; 
for i=1:length(orgNet.layers)
    if isfield(orgNet.layers{i}, 'learningRate')
        orgNet.layers{i}.learningRate = zeros(size(orgNet.layers{i}.learningRate));
    end
end
net.layers(end+1:end+length(orgNet.layers)) =  orgNet.layers;
end



% -------------------------------------------------------------------------
function opts=getAutoencoderOpts()
% -------------------------------------------------------------------------
%Initializes the options for the autoencoder training
%Make sure to change the paths to your data and models if needed

opts.nesterovUpdate = false;
opts.expDir = fullfile(vl_rootnn, 'data', 'autoencoder_with_classerror') ;
opts.dataDir = fullfile(vl_rootnn, 'data', 'dataset') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.orgNetPath= fullfile(opts.expDir, 'model-task-1.mat');
opts.dataLinks = 1; %1 if imdb.images.data contains images paths, 0 otherwise
opts.code_size=100;
opts.input_size= 9216;
opts.batchSize= 128;
opts.initial_encoder=[];
opts.errorType       = 'euclideanloss';
opts.errorFunction = 'euclideanloss' ;
opts.continue        = true;
opts.learningRate    = 1e-2;
opts.numEpochs       = 100; 
opts.imagenet_mean='imagenet_mean';
opts.imagenet_std='imagenet_std';
opts.compute_stats=false;
opts.alpha=1e-6;
opts.output_layer_id= 16;
opts.useGpu          = true;
opts.val             = [];
opts.weightDecay     = 5e-4;
opts.scale = 1 ;
opts.initBias = 0 ;
opts.weightInitMethod = 'xavierimproved' ;
opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.numFetchThreads = 12 ;
end


% -------------------------------------------------------------------------
function fn = getBatchFn(opts)
% -------------------------------------------------------------------------
fn = @(x,y) getBatch(x,y,opts) ;
end

% -------------------------------------------------------------------------
function varargout = getBatch(imdb, batch, opts)
% -------------------------------------------------------------------------
if opts.dataLinks
    im_links=imdb.images.data(batch);
input = [];
for i=1:length(im_links)
    im = load(im_links{i});
    input = cat(4, input, im.input);
end;
else
    input = imdb.images.data(:,:,:,batch);
end;
labels = imdb.images.labels(batch);
varargout = {input, labels} ;
end
