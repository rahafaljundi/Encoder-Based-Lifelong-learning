function res_out=multitask_forward_code_pool(layer, res_in, res_out)
%MULTITASK_FORWARD_CODE_POOL is a custom forward function for the first
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
switch lower(layer.mode)
    case 'normal'
        testMode = false ;
    case 'test'
        testMode = true ;
    otherwise
        error('Unknown mode ''%s''.', opts. mode) ;
end
for t=1: numel(layer.tasks)
    for t_layer=1:numel(layer.tasks{t}.layers)
        
        t_l=layer.tasks{t}.layers{t_layer};
        if(t_layer==1)
            task_input=res_in.x;           
            res_out.aux{t}.layers{t_layer}.x=res_in.x;
        else
            task_input=res_out.aux{t}.layers{t_layer}.x;
        end
        switch t_l.type
            case 'conv'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnconv(task_input, t_l.weights{1},t_l.weights{2}, ...
                    'pad', t_l.pad, 'stride', t_l.stride, ...
                    'dilate', t_l.dilate, ...
                    cudnn{:}) ;
            case 'pool'
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnpool(task_input, t_l.pool, ...
                    'pad', t_l.pad, 'stride', t_l.stride, ...
                    'method', t_l.method, ...
                    t_l.opts{:}, ...
                    cudnn{:}) ;
                
            case 'softmaxloss'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnsoftmaxloss(task_input, layer.class) ;
                
            case 'dropout'
               if testMode
                    res_out.aux{t}.layers{t_layer+1}.x = task_input ;
                else
                    [ res_out.aux{t}.layers{t_layer+1}.x , res_out.aux{t}.layers{t_layer+1}.mask] = vl_nndropout(task_input, 'rate', t_l.rate) ;
                end
            case 'softmaxlossdiff'
                res_out.aux{t}.layers{t_layer+1}.x= vl_nnsoftmaxdiff(task_input, layer.tasks_class{t}) ;
                
            case 'euclideanloss'
                
                res_out.aux{t}.layers{t_layer+1}.x= euclideanloss(task_input, layer.tasks_class{t});
            case 'intermediate_euclideanloss'
                res_out.aux{t}.layers{t_layer+1}.x= intermediate_euclideanloss(task_input, layer.tasks_class{t},t_l.alpha);                              
            case 'softmax'
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnsoftmax(task_input) ;

            case 'sigmoid'
                res_out.aux{t}.layers{t_layer+1}.x= vl_nnsigmoid(task_input) ;
            case 'relu'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnrelu(task_input,[],leak{:}) ;
                
            case 'en_reshape'
                res_out.aux{t}.layers{t_layer+1}.x= en_reshape(task_input) ;
            case 'standerize'
                res_out.aux{t}.layers{t_layer+1}.x= en_standarize(task_input,t_l.mu,t_l.std) ;
            case 'custom'
               t_l.testMode=testMode;
               t_l.tasks_targets=cell(numel(t_l.tasks));
               t_l.tasks_targets=layer.tasks_targets;
               t_l.class = layer.class;
                
                fc_res=res_out.aux{t}.layers{t_layer};
                
                
                res_out.aux{t}.layers{t_layer+1}=t_l.forward(t_l,  fc_res) ;
        end
        
    end
    
    
end
res_out.x=res_out.aux{end}.layers{end}.x;

end
