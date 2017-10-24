function res_out=multitask_forward_bef_fc8(layer, res_in)
%MULTITASK_FORWARD_BEF_FC8 is a custom forward function for the last
%fork in the model 
%This fork contains the specific task layers for each of  the tasks of the
%sequence
%
%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko,
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Rahaf Aljundi
%
% See the COPYING file.

cudnn = {'NoCuDNN'} ;
leak = {} ;
for t=1: numel(layer.tasks)
    for t_layer=1:numel(layer.tasks{t}.layers)
        t_l=layer.tasks{t}.layers{t_layer};
        if(t_layer==1)
            if (t<numel(layer.tasks))
                layer.tasks{t}.class=layer.tasks_targets{t};
            end
            task_input=res_in.x;
            res_out.aux{t}.layers{t_layer}.x=task_input;
        else
            task_input=res_out.aux{t}.layers{t_layer}.x;
        end
        switch t_l.type
            case 'conv'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnconv(task_input, t_l.weights{1},t_l.weights{2}, ...
                    'pad', t_l.pad, 'stride', t_l.stride, ...
                    'dilate', t_l.dilate, ...
                    cudnn{:}) ;
            case 'relu'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnrelu(task_input,[],leak{:}) ;
            case 'softmaxloss'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnsoftmaxloss(task_input, layer.class) ;
            case 'softmax'
                
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnsoftmax(task_input) ;
            case 'softmaxlossdiff'
                opts.mode='MI';
                opts.temperature=2;
                res_out.aux{t}.layers{t_layer+1}.x = vl_nnsoftmaxdiff(task_input, layer.tasks{t}.class,[],opts) ;
            case 'dropout'
                if layer.testMode
                    res_out.aux{t}.layers{t_layer+1}.x = task_input ;
                else
                    [ res_out.aux{t}.layers{t_layer+1}.x , res_out.aux{t}.layers{t_layer+1}.mask] = vl_nndropout(task_input, 'rate', t_l.rate) ;
                end
        end
        
    end
    
    
end
res_out.x=res_out.aux{end}.layers{end}.x;


end
