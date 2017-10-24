function setup_LwF_with_encoder(MatconvnetPath, compile, useGPU)
%Setups matconvnet inorder to use Encoder Based Lifelong Learning codes
%
%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko,
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Amal Rannen Triki
%
% See the COPYING file.

if nargin < 1
    MatconvnetPath = './';
end;
if nargin < 2
    compile = 1;
end;
if nargin < 3
    useGPU = 1;
end;
    
addpath(genpath('..'));
cd(MatconvnetPath);
run  matlab/vl_setupnn
if compile
    if useGPU
        vl_compilenn('enableGpu', true);
    else
        vl_compilenn('enableGpu', false);
    end;
end;
