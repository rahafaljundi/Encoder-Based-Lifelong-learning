function Y = intermediate_euclideanloss(X, c, alpha,dzdy)
%INTERMEDIATE_EUCLIDEANLOSS  computes euclidean distance between X and c
%multiplied by alpha, but just forward X to the next layers.
%This function is used as loss for autoencoder training, and for the code
%loss in our global model.
%
%For more details about the model, see A. Rannen Triki, R. Aljundi, M. B. Blaschko,
%and T. Tuytelaars, Encoder Based Lifelong Learning. ICCV 2017
%
% Author: Rahaf Aljundi
%
% See the COPYING file.

if(~exist('alpha','var'))
alpha = 1e-6;
end
assert(numel(X) == numel(c));

if nargin == 3 || (nargin == 4 && isempty(dzdy))    
    Y = 1 / 2 * (1/size(X,4))*sum(subsref((X - c) .^ 2, substruct('()', {':'})));
    Y=X;
elseif nargin == 4 && ~isempty(dzdy)
    Y = dzdy + alpha*(X - c);     
end

end
