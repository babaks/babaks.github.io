function tree = makeTree(allClass);

% This code gets a matrix of classes (called allClass) and returns a tree
% structure. allClass is an n*depth matrix, where "n" is the sample size 
% and "depth" is the number of tree levels. The first column contains the node
% labels of for the first level of the tree (starting from the root). These
% labels on the tree increase from left to right. They don't have to be sequential
% and they don't have to start from one, but they should be bigger than
% zero since here zero means that there is no additional split in that
% node. For example, The following matrix presents a sample of 10 cases,
% whose classes have a three-level hierarchical structure.

% allClass =
% 
%      1     1     2
%      1     1     3
%      2     3     5
%      2     4     7
%      3     5     8
%      3     6     0
%      1     1     2
%      3     6     0
%      1     2     0
%      3     6     0
     
% Based on this matrix, there are 3 nodes in the first level, 6 nodes in the 
% second level and 2 nodes in the last level.  The resulting tree is:

% tree =                                                ______|_____
%                                                   ___|__   _|_   _|_
%      1     1     1     2     2     3     3      _|_     | |   | |   |
%      1     1     2     3     4     5     6     |   |
%      1     2     0     0     0     0     0      
     
% if you have the class labels in a comma-delimited text file (i.e., 
% allClass.dat), first read them in using: 
% allClass = dlmread('allClass.dat');

[u1, u2, u3] = unique(allClass, 'rows');

initTree = u1';

[dim1, dim2] = size(u1);

for i = 1:dim1 - 1
    temp =  u1(i, u1(i, :)~=0);
    ind = length(temp);
    a = logical (u1(i, 1:ind) == u1(i+1, 1:ind)) + 0;
    if prod(a)
        u1(i, ind+1) = .5;
    end
end

newU1 = u1;
for i = 1:dim1
    ind =  length(u1(i, u1(i, :)~=0));
    a = newU1(i, :);
    for j = (ind-1):-1:1
        temp = newU1;
        temp(i, :) = [];
        b = ismember(temp(:, 1:j), a(:, 1:j), 'rows');
        if sum(b) == 0
            newU1(i, (j+1):end) = 0;
        end
    end
end


for j = 1:dim2 - 1
    count = 0;
    [temp1, temp2, temp3] = unique(newU1(:, j));
    for i = 1:length(temp1)
        ind = find(newU1(:, j) == temp1(i));
        temp4 = unique(newU1(ind, j+1));
        if length(temp4) == 1
            newU1(ind, j+1:end-1) = newU1(ind, j+2:end);
            newU1(ind, end) = 0;
        end
    end
end


[temp1, temp2, temp3] = unique(newU1(:, 1));
temp4 = [1:length(temp1)]';
newU1(:, 1) = temp4(temp3);

for j = 1:dim2 - 1
    count = 0;
    [temp1, temp2, temp3] = unique(newU1(:, j));
    for i = 1:length(temp1)
        if temp1(i) == 0
            continue;
        else
            temp4 = newU1(newU1(:, j) == temp1(i), j+1);
            [temp5, temp6, temp7] = unique(temp4);
            if length(temp5) == 1 & temp5 == 0
                continue;
            end
            temp8 = [1:length(temp5(temp5~=0))]' + count;
            if temp5(1) == 0
                temp8 = [0; temp8];
            end
            newU1(newU1(:, j) == temp1(i), j+1) = temp8(temp7);
            count = temp8(end);
        end
    end
end
        
     
tree = newU1';
