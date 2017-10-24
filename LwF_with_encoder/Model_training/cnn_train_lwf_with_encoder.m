function [net, stats] = cnn_train_lwf_with_encoder(net, imdb, getBatch, varargin)
%CNN_TRAIN_LWF_WITH_ENCODER is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN with the Encoder Based
%    Lifelong Learning method after preparing the model(See functions under
%    Model_prepapration).
%    It can be used with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).
%
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


opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.saveMomentum = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;
opts.encoder_fork_indx=16;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

net = vl_simplenn_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    for i=1:numel(net.layers)
        if(strcmp(net.layers{i}.type,'custom'))
            net.layers{i}=initilize_custom(net.layers{i});
        else
            J = numel(net.layers{i}.weights) ;
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J) ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J) ;
            end
        end
    end
end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
    switch opts.errorFunction
        case 'none'
            opts.errorFunction = @error_none ;
            hasError = false ;
        case 'multiclass'
            opts.errorFunction = @error_multiclass ;
            if isempty(opts.errorLabels), opts.errorLabels = {'top1err', 'top5err'} ; end
        case 'binary'
            opts.errorFunction = @error_binary ;
            if isempty(opts.errorLabels), opts.errorLabels = {'binerr'} ; end
        otherwise
            error('Unknown error function ''%s''.', opts.errorFunction) ;
    end
end

state.getBatch = getBatch ;
stats = [] ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net, state, stats] = loadState(modelPath(start)) ;
else
    state = [] ;
end

for epoch=start+1:opts.numEpochs
    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.
    
    rng(epoch + opts.randomSeed) ;
    prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.
    params = opts ;
    params.epoch = epoch ;
    params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    params.val = opts.val(randperm(numel(opts.val))) ;
    params.imdb = imdb ;
    params.getBatch = getBatch ;
    
    if numel(params.gpus) <= 1
        [net, state] = processEpoch(net, state, params, 'train') ;
        [net, state] = processEpoch(net, state, params, 'val') ;
        if ~evaluateMode
            saveState(modelPath(epoch), net, state) ;
        end
        lastStats = state.stats ;
    else
        spmd
            [net, state] = processEpoch(net, state, params, 'train') ;
            [net, state] = processEpoch(net, state, params, 'val') ;
            if labindex == 1 && ~evaluateMode
                saveState(modelPath(epoch), net, state) ;
            end
            lastStats = state.stats ;
        end
        lastStats = accumulateStats(lastStats) ;
    end
    
    stats.train(epoch) = lastStats.train ;
    stats.val(epoch) = lastStats.val ;
    clear lastStats ;
    saveStats(modelPath(epoch), stats) ;
    
    if params.plotStatistics
        switchFigure(1) ; clf ;
        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ;
            values = zeros(0, epoch) ;
            leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            subplot(1,numel(plots),find(strcmp(p,plots))) ;
            plot(1:epoch, values','o-') ;
            xlabel('epoch') ;
            title(p) ;
            %legend(leg{:}) ;
            grid on ;
        end
        drawnow ;
        print(1, modelFigPath, '-dpdf') ;
    end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function err = error_multiclass(params, labels, res)
% -------------------------------------------------------------------------
predictions=gather(res(end).aux{end}.layers{end}.aux{end}.layers{end-1}.x);
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
    labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
    % if there is a second channel in labels, used it as weights
    mass = mass .* labels(:,:,2,:) ;
    labels(:,:,2,:) = [] ;
end

m = min(5, size(predictions,3)) ;

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:m,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binary(params, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(params, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0

if isempty(state) || isempty(state.momentum)
    for i = 1:numel(net.layers)
        if(strcmp(net.layers{i}.type,'custom'))
            state.momentum{i}=initilize_momentum(net.layers{i});
        end
        if isfield(net.layers{i},'weights')
            for j = 1:numel(net.layers{i}.weights)
                state.momentum{i}{j} = 0 ;
            end
        end
    end
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
    net = vl_simplenn_move_lwf(net, 'gpu') ;
    for i = 1:numel(state.momentum)
        if(isstruct(state.momentum{i}))
            state.momentum{i}=move_momentum(state.momentum{i},@gpuArray);
        else
            for j = 1:numel(state.momentum{i})
                state.momentum{i}{j} = gpuArray(state.momentum{i}{j}) ;
            end
        end
    end
end
if numGpus > 1
    parserv = ParameterServer(params.parameterServer) ;
    vl_simplenn_start_parserv(net, parserv) ;
else
    parserv = [] ;
end

% profile
if params.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

subset = params.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;

start = tic ;
for t=1:params.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, params.epoch, ...
        fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    batchSize = min(params.batchSize, numel(subset) - t + 1) ;
    
    for s=1:params.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+params.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        [im, labels,tasks_recs,tasks_targets] = params.getBatch(params.imdb, batch) ;
        if params.prefetch
            if s == params.numSubBatches
                batchStart = t + (labindex-1) + params.batchSize ;
                batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
            params.getBatch(params.imdb, nextBatch) ;
        end
        
        if numGpus >= 1
            im = gpuArray(im) ;
            for ts=1:numel(tasks_recs)
                tasks_recs{ts}=gpuArray(tasks_recs{ts});
            end
        end
        
        if strcmp(mode, 'train')
            dzdy = 1 ;
            evalMode = 'normal' ;
        else
            dzdy = [] ;
            evalMode = 'test' ;
        end
        net.layers{end}.class = labels ;
        net.layers{end}.tasks_class = tasks_recs ;
        net.layers{end}.tasks_targets = tasks_targets;
        
        net.layers{end}.mode=evalMode;
        res = vl_simplenn(net, im, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', params.conserveMemory, ...
            'backPropDepth', params.backPropDepth, ...
            'sync', params.sync, ...
            'cudnn', params.cudnn, ...
            'parameterServer', parserv, ...
            'holdOn', s < params.numSubBatches) ;
        % accumulate errors
        error = sum([error, [...
            sum(double(gather(res(end).x))) ;
            reshape(params.errorFunction(params, labels, res),[],1) ; ]],2) ;
    end
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(parserv), parserv.sync() ; end
        [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv) ;
    end
    
    % get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = extractStats(net, params, error / num) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == 3*params.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s: %.3f', f, stats.(f)) ;
    end
    fprintf('\n') ;
    for ts=1:numel(net.layers{end}.tasks)-1
        fprintf(' [%f/ task number %d  error]', (res(end).aux{end}.layers{end}.aux{ts}.layers{end}.x/ numel(batch)), ts);
    end
    fprintf('\n') ;
    if strcmp(mode, 'train') && params.plotDiagnostics
        switchFigure(2) ; clf ;
        diagn = [res.stats] ;
        diagnvar = horzcat(diagn.variation) ;
        diagnpow = horzcat(diagn.power) ;
        subplot(2,2,1) ; barh(diagnvar) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnvar), ...
            'YTickLabel',horzcat(diagn.label), ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1], ...
            'XTick', 10.^(-5:1)) ;
        grid on ;
        subplot(2,2,2) ; barh(sqrt(diagnpow)) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnpow), ...
            'YTickLabel',{diagn.powerLabel}, ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1e5], ...
            'XTick', 10.^(-5:5)) ;
        grid on ;
        subplot(2,2,3); plot(squeeze(res(end-1).x)) ;
        drawnow ;
    end
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
    if numGpus <= 1
        state.prof.(mode) = profile('info') ;
        profile off ;
    else
        state.prof.(mode) = mpiprofile('info');
        mpiprofile off ;
    end
end
if ~params.saveMomentum
    state.momentum = [] ;
else
    
    for i = 1:numel(state.momentum)
        if(isstruct(state.momentum{i}))
            state.momentum{i}=move_momentum(state.momentum{i},@gather);
        else
            for j = 1:numel(state.momentum{i})
                state.momentum{i}{j} = gather(state.momentum{i}{j}) ;
            end
        end
    end
end

net = vl_simplenn_move_lwf(net, 'cpu') ;

% -------------------------------------------------------------------------
function [net, res, state] = accumulateGradients(net, res, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;
%===========================LWF-Auto:start============================
%take care of the last fully connected layers
fc_for_layer=net.layers{end}.tasks{end}.layers{end};
fc_res=res(end-1).aux{end}.layers{end-1};
this_state=state.momentum{end}.tasks{end}.layers{end};
for t=1:numel(fc_res.aux)
    %only one fully connected layer
    for j=numel(fc_res.aux{t}.layers{1}.dzdw):-1:1% has to be dealt with
        thisDecay = params.weightDecay *fc_for_layer.tasks{t}.layers{1}.weightDecay(j) ;
        thisLR =  params.learningRate  *fc_for_layer.tasks{t}.layers{1}.learningRate(j) ;
        this_state.tasks{t}.layers{1}{j}=params.momentum * this_state.tasks{t}.layers{1}{j} ...
            - thisDecay * fc_for_layer.tasks{t}.layers{1}.weights{j} ...
            - (1 / batchSize) * fc_res.aux{t}.layers{1}.dzdw{j} ;
        fc_for_layer.tasks{t}.layers{1}.weights{j} = fc_for_layer.tasks{t}.layers{1}.weights{j} + thisLR *this_state.tasks{t}.layers{1}{j};
        
    end
end
state.momentum{end}.tasks{end}.layers{end}=this_state;
net.layers{end}.tasks{end}.layers{end}=fc_for_layer;

%first accumulate the gradients from the last multi task layer
for t=numel(net.layers{end}.tasks):-1:1
    %Here I can add the AUtoencoder and the LWF
    %freeze the autoencoder
    %just the first layer is convolutional
    
    for t_layer=numel(net.layers{end}.tasks{t}.layers):-1:1
        if(isfield(res(end-1).aux{t}.layers{t_layer},'dzdw'))%for the soft max layers
            for j=numel(res(end-1).aux{t}.layers{t_layer}.dzdw):-1:1% has to be dealt with
                thisDecay = params.weightDecay * net.layers{end}.tasks{t}.layers{t_layer}.weightDecay(j) ;
                thisLR =  params.learningRate  * net.layers{end}.tasks{t}.layers{t_layer}.learningRate(j) ;
                this_batch_size=size(res(end-1).aux{t}.layers{t_layer}.dzdx,4);
                if(thisLR>0)
                    state.momentum{end}.tasks{t}.layers{t_layer}{j}=params.momentum * state.momentum{end}.tasks{t}.layers{t_layer}{j} ...
                        - thisDecay * net.layers{end}.tasks{t}.layers{t_layer}.weights{j} ...
                        - (1 / this_batch_size) * res(end-1).aux{t}.layers{t_layer}.dzdw{j} ;
                    net.layers{end}.tasks{t}.layers{t_layer}.weights{j} = net.layers{end}.tasks{t}.layers{t_layer}.weights{j} + thisLR *state.momentum{end}.tasks{t}.layers{t_layer}{j} ;
                end
            end
        end
    end
end

for l=numel(net.layers)-1:-1:1
    for j=numel(res(l).dzdw):-1:1
        
        if ~isempty(parserv)
            tag = sprintf('l%d_%d',l,j) ;
            parDer = parserv.pull(tag) ;
        else
            parDer = res(l).dzdw{j}  ;
        end
        
        if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
            % special case for learning bnorm moments
            thisLR = net.layers{l}.learningRate(j) ;
            net.layers{l}.weights{j} = vl_taccum(...
                1 - thisLR, ...
                net.layers{l}.weights{j}, ...
                thisLR / batchSize, ...
                parDer) ;
        else
            % Standard gradient training.
            thisDecay = params.weightDecay * net.layers{l}.weightDecay(j) ;
            thisLR = params.learningRate * net.layers{l}.learningRate(j) ;
            
            
            
            if thisLR>0 || thisDecay>0
                % Normalize gradient and incorporate weight decay.
                parDer = vl_taccum(1/batchSize, parDer, ...
                    thisDecay, net.layers{l}.weights{j}) ;
                
                % Update momentum.
                state.momentum{l}{j} = vl_taccum(...
                    params.momentum, state.momentum{l}{j}, ...
                    -1, parDer) ;
                
                % Nesterov update (aka one step ahead).
                if params.nesterovUpdate
                    delta = vl_taccum(...
                        params.momentum, state.momentum{l}{j}, ...
                        -1, parDer) ;
                else
                    delta = state.momentum{l}{j} ;
                end
                
                % Update parameters.
                net.layers{l}.weights{j} = vl_taccum(...
                    1, net.layers{l}.weights{j}, ...
                    thisLR, delta) ;
            end
        end
        
        % if requested, collect some useful stats for debugging
        if params.plotDiagnostics
            variation = [] ;
            label = '' ;
            switch net.layers{l}.type
                case {'conv','convt'}
                    variation = thisLR * mean(abs(state.momentum{l}{j}(:))) ;
                    power = mean(res(l+1).x(:).^2) ;
                    if j == 1 % fiters
                        base = mean(net.layers{l}.weights{j}(:).^2) ;
                        label = 'filters' ;
                    else % biases
                        base = sqrt(power) ;
                        label = 'biases' ;
                    end
                    variation = variation / base ;
                    label = sprintf('%s_%s', net.layers{l}.name, label) ;
            end
            res(l).stats.variation(j) = variation ;
            res(l).stats.power = power ;
            res(l).stats.powerLabel = net.layers{l}.name ;
            res(l).stats.label{j} = label ;
        end
    end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------
for s = {'train', 'val'}
    s = char(s) ;
    total = 0 ;
    
    % initialize stats stucture with same fields and same order as
    % stats_{1}
    stats__ = stats_{1} ;
    names = fieldnames(stats__.(s))' ;
    values = zeros(1, numel(names)) ;
    fields = cat(1, names, num2cell(values)) ;
    stats.(s) = struct(fields{:}) ;
    
    for g = 1:numel(stats_)
        stats__ = stats_{g} ;
        num__ = stats__.(s).num ;
        total = total + num__ ;
        
        for f = setdiff(fieldnames(stats__.(s))', 'num')
            f = char(f) ;
            stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;
            
            if g == numel(stats_)
                stats.(s).(f) = stats.(s).(f) / total ;
            end
        end
    end
    stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(net, params, errors)
% -------------------------------------------------------------------------
stats.objective = errors(1) ;
for i = 1:numel(params.errorLabels)
    stats.(params.errorLabels{i}) = errors(i+1) ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net, state)
% -------------------------------------------------------------------------
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
    save(fileName, 'stats', '-append') ;
else
    save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = vl_simplenn_tidy(net) ;
if isempty(whos('stats'))
    error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
    try
        set(0,'CurrentFigure',n) ;
    catch
        figure(n) ;
    end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
%clear vl_tmove vl_imreadjpeg ;
disp('Clearing mex files') ;
clear mex ;
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(params, cold)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate') ;
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(pool) ;
    end
    pool = gcp('nocreate') ;
    if isempty(pool)
        parpool('local', numGpus) ;
        cold = true ;
    end
end
if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename) ;
    clearMex() ;
    if numGpus == 1
        disp(gpuDevice(params.gpus)) ;
    else
        spmd
            clearMex() ;
            disp(gpuDevice(params.gpus(labindex))) ;
        end
    end
end

% -------------------------------------------------------------------------
function layer=initilize_custom(layer)
% -------------------------------------------------------------------------
%Initializes custom layer (= branches that contain encoders and task layers
%of previous task, and the task layer of current task)
for t=1: numel(layer.tasks)
    %each task has at least two layers: one conv and one
    %softmaxloss or softmaxlossdiff
    tasks_layers= layer.tasks{t}.layers;
    for t_layer=1:numel(tasks_layers)
        if isfield(tasks_layers{t_layer}, 'weights')
            J = numel(tasks_layers{t_layer}.weights) ;
            if ~isfield(tasks_layers{t_layer}, 'learningRate')
                tasks_layers{t_layer}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(tasks_layers{t_layer}, 'weightDecay')
                tasks_layers{t_layer}.weightDecay = ones(1, J, 'single') ;
            end
        end
        if(strcmp(tasks_layers{t_layer}.type,'custom'))
            tasks_layers{t_layer}=initilize_custom(tasks_layers{t_layer});
        end
        
    end
    layer.tasks{t}.layers=tasks_layers;
    clear tasks_layers;
end

% -------------------------------------------------------------------------
function state=move_momentum(state,fn)
% -------------------------------------------------------------------------
%Reccursive call of move_momentum to deal with all the branches of the
%model
for t= 1:numel(state.tasks)
    for i = 1:numel(state.tasks{t}.layers)
        if(isstruct(state.tasks{t}.layers{i}))
            state.tasks{t}.layers{i}=move_momentum( state.tasks{t}.layers{i},fn);
        else
            for j = 1:numel(state.tasks{t}.layers{i})
                state.tasks{t}.layers{i}{j}  = fn(state.tasks{t}.layers{i}{j} ) ;
            end
        end
    end
end

% -------------------------------------------------------------------------
function momentum=initilize_momentum(layer)
% -------------------------------------------------------------------------
%Initializes the momentum for the custom layer (= branches that contain 
%encoders and task layers of previous task, and the task layer of current 
%task)
for t=1: numel(layer.tasks)
    tasks_layers= layer.tasks{t}.layers;
    
    for t_layer=1:numel(tasks_layers)
        if(strcmp(tasks_layers{t_layer}.type,'custom'))
            temp.tasks{t}.layers{t_layer}= initilize_momentum(tasks_layers{t_layer});
        end
        if isfield(tasks_layers{t_layer},'weights')
            J = numel(tasks_layers{t_layer}.weights) ;
            for j=1:J
                temp.tasks{t}.layers{t_layer}{j} = 0 ;
            end
        end
    end
end
momentum=temp;

