function out = en_standarize(x,mu,std,dzdy)
% Standarizes the layer based on the layer mean and standerd deviation
% Author: Rahaf Aljundi
%
% See the COPYING file.

initial_dimensions=size(x);

temp=uint8(mu);
x=reshape(x,size(temp,1),size(temp,2),size(temp,3),[]);
x= bsxfun(@minus,x,mu);
y= bsxfun(@rdivide, x,std);

if nargin <= 3 || isempty(dzdy)
  out = y ;
else
  out = dzdy./std ;
  out=reshape(out,initial_dimensions);
end
