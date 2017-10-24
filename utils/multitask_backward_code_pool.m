function res_in=multitask_backward_code_pool(layer, res_in, res_out)
%MULTITASK_BACKWARD_CODE_POOL is a custom backward function for the first
%fork in the model 
%This fork contains the autoencoders + the task layers
%The task layers have another fork (custom layer) in the end.
%
%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko,
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Rahaf Aljundi
%
% See the COPYING file.


cudnn = {'NoCuDNN'} ;
leak = {} ;
res_in.dzdx=0;

switch lower(layer.mode)
    case 'normal'
        testMode = false ;
    case 'test'
        testMode = true ;
    otherwise
        error('Unknown mode ''%s''.', opts. mode) ;
end

for t= numel(layer.tasks):-1:1
    max_number_of_layers=numel(res_out.aux{t}.layers);
    res_in.aux{t}.layers{max_number_of_layers}.dzdx=res_out.dzdx;
    for t_layer=numel(layer.tasks{t}.layers):-1:1
        t_l=layer.tasks{t}.layers{t_layer};
        switch t_l.type
            case 'conv'
                [res_in.aux{t}.layers{t_layer}.dzdx, res_in.aux{t}.layers{t_layer}.dzdw{1}, res_in.aux{t}.layers{t_layer}.dzdw{2}] = ...
                    vl_nnconv( res_out.aux{t}.layers{t_layer}.x , t_l.weights{1}, t_l.weights{2}, ...
                    res_in.aux{t}.layers{t_layer+1}.dzdx, ...
                    'pad', t_l.pad, 'stride', t_l.stride, ...
                    cudnn{:}) ;
            case 'pool'
                res_in.aux{t}.layers{t_layer}.dzdx= vl_nnpool(res_out.aux{t}.layers{t_layer}.x , t_l.pool, res_in.aux{t}.layers{t_layer+1}.dzdx, ...
                    'pad', t_l.pad, 'stride', t_l.stride, ...
                    'method', t_l.method, ...
                    t_l.opts{:}, ...
                    cudnn{:}) ;
            case 'softmaxloss'
                res_in.aux{t}.layers{t_layer}.dzdx = vl_nnsoftmaxloss(res_out.aux{t}.layers{t_layer}.x, layer.class, res_in.aux{t}.layers{t_layer+1}.dzdx) ;
                
            case 'softmaxlossdiff'
                res_in.aux{t}.layers{t_layer}.dzdx = vl_nnsoftmaxdiff(res_out.aux{t}.layers{t_layer}.x, layer.tasks_class{t}, res_in.aux{t}.layers{t_layer+1}.dzdx) ;
            case 'euclideanloss'
                res_in.aux{t}.layers{t_layer}.dzdx  = euclideanloss(res_out.aux{t}.layers{t_layer}.x, layer.tasks_class{t},  res_in.aux{t}.layers{t_layer+1}.dzdx);
            case 'intermediate_euclideanloss'
                res_in.aux{t}.layers{t_layer}.dzdx  = intermediate_euclideanloss(res_out.aux{t}.layers{t_layer}.x, layer.tasks_class{t},  t_l.alpha,res_in.aux{t}.layers{t_layer+1}.dzdx);              
            case 'dropout'
                if testMode
                    res_in.aux{t}.layers{t_layer}.dzdx = res_in.aux{t}.layers{t_layer+1}.dzdx ;
                else
                    res_in.aux{t}.layers{t_layer}.dzdx = vl_nndropout(res_out.aux{t}.layers{t_layer}.x, res_in.aux{t}.layers{t_layer+1}.dzdx, ...
                        'mask',res_out.aux{t}.layers{t_layer+1}.mask) ;
                end
            case 'sigmoid'
                res_in.aux{t}.layers{t_layer}.dzdx= vl_nnsigmoid(res_out.aux{t}.layers{t_layer}.x,res_in.aux{t}.layers{t_layer+1}.dzdx) ;
            case 'relu'
                res_in.aux{t}.layers{t_layer}.dzdx = vl_nnrelu(res_out.aux{t}.layers{t_layer}.x,res_in.aux{t}.layers{t_layer+1}.dzdx, leak{:}) ;            
            case 'en_reshape'
                res_in.aux{t}.layers{t_layer}.dzdx= en_reshape(res_out.aux{t}.layers{t_layer}.x,res_in.aux{t}.layers{t_layer+1}.dzdx) ;
            case 'standerize'
                res_in.aux{t}.layers{t_layer}.dzdx= en_standarize(res_out.aux{t}.layers{t_layer}.x,t_l.mu,t_l.std,res_in.aux{t}.layers{t_layer+1}.dzdx) ;
            case 'custom'
                t_l.testMode=testMode;
                t_l.tasks_targets=layer.tasks_targets;                
                t_l.class = layer.class;              
                res_in.aux{t}.layers{t_layer}=t_l.backward(t_l,res_out.aux{t}.layers{t_layer},res_out.aux{t}.layers{t_layer+1}) ;     
        end
    end
   
    %sumup the gradients from all the tasks
    if(~isfield(layer.tasks{t}.layers{1},'gr_weight'))
        layer.tasks{t}.layers{1}.gr_weight=1;
    res_in.dzdx=res_in.dzdx+ layer.tasks{t}.layers{1}.gr_weight*res_in.aux{t}.layers{1}.dzdx;
    end
end
 clear res_out

