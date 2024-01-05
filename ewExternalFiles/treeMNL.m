function treeMNL(numberOfIterations, burnIn)
global tree nLeaf e leapFrog;

% ***************** General Instruction

% To run this program, you first need to create a matlab dataset called 
% "data.mat" and put it in the same folder as this file. The dataset should
% include the following matrices: "train", "test", "rTrain" and "rTest"
% which are the training set, the test set, the response values for the
% training cases and the response values for the test cases respectively.
% "train" and "test" contain the explanatory variables. Later, a vector of 
% 1's will be added to these datasets to account for the intercept. 
% Response values are class labels, which start from 1 and increments by 1.

% The file data.mat should also include another matrix called "tree", which 
% defines the hierarchical structure of classes. For this matrix  
% dimension=depth*nLeaf, where "depth" is the number of levels and "nLeaf" 
% is the number of end nodes in the tree structure. Starting from the top 
% level, each row corresponds to one level of the hierarchy. Each column of
% this matrix represents one class. At each row, classes with a common node 
% are given the same number. When the branches that lead to class "j" end at 
% level "k" of the hierarchy, all elements in the column j with row numbers 
% biger than k are set to zero. This way, the program knows that for each 
% covariate there should be only k parameters corresponding to class j.

% To illustrate this, we can look at the tree used for Figure 3. (the complex 
% hierarchy) in the paper. The tree matrix for this hierarchy is: 
% tree = [[1, 1, 1, 1, 1, 2, 2, 3]; ...
%        [ 1, 1, 2, 2, 2, 3, 4, 0]; ...
%        [ 1, 2, 3, 4, 5, 0, 0, 0]];

% When the file is ready, you can run the program by entering the command 
% treeMNL(N, n), where "N" is the number
% of MCMC iterations, and "n" is the number of discarded MCMC samples
% (i.e., burn-in).
 

% The current values of each MCMC iteration will be written in several files.
% These files are treeBeta.dat (regression parameters with the interecepts if
% the first row), treeArdSigma.dat (ARD parameters), and treeTau.dat (scale 
% parameters). Since the beta matrix is three dimensional (i.e., the 3rd 
% dimension corresponds to the levels of the tree), MATLAB writes the
% elements of beta(:, :, 1) first, beta(:, :, 2) second and ...
% The ARD and scale hyperparameters are only updated after
% the 10th iteration. This is to make sure that beta's are updated first. 
% Note that these files are only used as the initial values for future runs. 
% The trace of MCMC samples are written in another file called sampTree.mat, 
% which saves the values at iterations 10, 20, 30, and so forth. The performance
% of the model, measured in terms of average log-probability and accuracy, for
% test cases is wirtten in a file called "resTreeTest". 


% *************************************

% ***************** Fixed parameters

abSigmaInt = [.5, 1]; % Intercept ~ Gamma(.5, 1)
abSigma = [1, 10]; % ARD parameter: \rho ~ Gamma(1, 10)
abTau = [.5, 100]; % Scale parameter" \tau ~ Gamma(.5, 20)

leapFrog = 500; % Number of leap-frogs in Hamiltonian dynamics
e = 0.01; % Stepsize, \epsilon, for regression parameters

nSkip = 1; % Number of MCMC samples skipped before saving the current
% values. This is just for evaluating convergances. Prediction is based
% on all samples after the initial discard.

% *************************************

% ***************** Preparing the data

% Here, I load the NIPS dataset, which is stroed in the file data.mat.
load data.mat;

% Here, nTrain, nTest and inputNum are variables that store the number of
% training case, the number of test cases and the number of covariates.
[nTrain, inputNum] = size(train);
[nTest,  inputNum] = size(test);

% At this part, I create a matrix of dummy variables to represent all
% possible classes.
allR = [rTrain; rTest];
allRDummy = dummyvar(allR);
rTrainDummy = allRDummy(1:nTrain, :);
rTestDummy  = allRDummy(nTrain+1:end, :);


% I add a column of 1's to the begining of each dataset. This is used as
% the multiplier for the intercept parameter.
train = [ones(nTrain, 1), train];
test  = [ones(nTest, 1),  test ];

% For simplicity, I use "d" and "r" for training data and training
% response variable.
d = train;
r = rTrainDummy;

% As mentioned above, depth and nLeaf are the number of levels and the
% number of end nodes in the tree structure.
[depth, nLeaf] = size(tree);

% This is another tree created based on the orginal tree and is used for
% simplifying the book-keeping of hyperparameters. The new tree has two
% extra levels, one at the top representing the root node (with all
% elements equal to one since all classes share the root) and one level at
% the end with values corresponding to classes.
tempTree = tree;
tree = [ones(1, nLeaf); tempTree; 1:nLeaf];
[depth, nLeaf] = size(tree);

% ***************** Initial values

% These are the initial values, if there are already inital values from the
% last runs, which are stored in the same directory, these values will be
% ignored.

% Number of regression parameters including the intercept.
pTrain  = inputNum  + 1;

% The first row of regression parameters, b, is for the intercept. Note
% that I use depth-2 here since the new tree created above has two extra
% levels for simplifying the book-keeping.
b = zeros(pTrain, nLeaf, depth-2);

% This is the mean of regression parameters. Here, mu0 is
% fixed and will not be updated.
mu0 = zeros(pTrain, nLeaf, depth-2);

% The variance of regression parameters (except for the intercept parameter)
% is the prduct of the following two hyperparameters.
ardSigma0 = ones(pTrain, nLeaf, depth-2);
tau = ones(1, 100);

% The number of previous MCMC samples and the corrsponding probability
% distribution. If there are previous samples, these will be changed later.
prevProbTest = 0;
prevNSamp = 0;


% This part checks whether there are previous samples available which can
% be used as starting point. These are used for running MCMC algorithms in
% more than one session.
msgId = '';
try 
    readBeta = dlmread('treeBeta.dat');
     readArdSigma = dlmread('treeArdSigma.dat');
catch
     [errmsg, msgId] = lasterr;
     msgId = lower(msgId);
end


% If there are initial values, this part of the code reads them in.
if isempty(msgId)
    b = reshape(dlmread('treeBeta.dat'), [pTrain, nLeaf, depth-2]);
    ardSigma0 = reshape(dlmread('treeArdSigma.dat'), [pTrain, nLeaf, depth-2]);
    tau = dlmread('treeTau.dat');
end


prevSampSize = 0;
msgId = '';
try 
    load sampTree;
catch
    [errmsg, msgId] = lasterr;
    msgId = lower(msgId);
    if msgId ~= 'matlab:load:couldnotreadfile'
        load sampTree;
        prevSampSize = length(sampTree);
             
        prevProbTest = dlmread('treeProb.dat');
        prevNSamp = dlmread('treeNSamp.dat');

    end
end

 
% This part finds the related cases and classes for each node.
trainTarget = zeros(nTrain, depth-1);
for i = depth:-1:2
        for j = 1:nLeaf
            trainTarget(rTrain == tree(depth, j) , i-1) = tree(i-1, j);
        end
end

testTarget = zeros(nTest, depth-1);
for i = depth:-1:2
        for j = 1:nLeaf
            testTarget(rTest == tree(depth, j) , i-1) = tree(i-1, j);
        end
end
       
 
[dim1, dim2, dim3] = size(b);

 for k = 1:dim3
    u1 = unique(trainTarget(:, k));
    
    for j = 1:length(u1)

    [uniqueVal, relatedInd, mapInd] = unique(tree(k+1, tree(k, :) == u1(j)));
        
    isThereZero = logical(uniqueVal(1) == 0);
    
    
    rTemp = (trainTarget(trainTarget(:, k)==u1(j), k+1));
 
    counter = 1;
    for i= uniqueVal(uniqueVal~=0)
        relatedR(k).a(j).b(:, counter) = logical(rTemp==i);
        counter = counter+1;
    end
    
    relatedNClass(k).a(j).b = length(uniqueVal(uniqueVal~=0));

    relatedD(k).a(j).b = d(trainTarget(:, k)==u1(j), :);

    end
 end
    
   

newProbTest = zeros(nTest, nLeaf);
countP = 0;
counter = 0;
startTime = cputime;


% *************************************

% ***************** MCMC algorithm

for iter = 1:numberOfIterations

iter
    
probTest = ones(nTest, nLeaf, depth-2);

[dim1, dim2, dim3] = size(b);

countTau = 0;
tempTau = [];    

for k = 1:dim3
    u1 = unique(trainTarget(:, k));
    
    for j = 1:length(u1)

        if u1(j) == 0
            b(:, find(tree(k, :) == u1(j)), k) = 0;
            sigma0(:, find(tree(k, :) == u1(j)), k) =1;
            continue;
        end

        [uniqueVal, relatedInd, mapInd] = unique(tree(k+1, tree(k, :) == u1(j)));

        if length(uniqueVal) == 1
            b(:, find(tree(k, :) == u1(j)), k) =0;
            sigma0(:, find(tree(k, :) == u1(j)), k) =1;
            continue;
        end
        
 
    tempMu = mu0(:, find(tree(k, :) == u1(j)), k);
    relatedMu = tempMu(:, relatedInd(uniqueVal ~= 0));

    tempArdSigma = ardSigma0(:, find(tree(k, :) == u1(j)), k);
    relatedArdSigma = tempArdSigma(:, relatedInd(uniqueVal ~= 0));    
    countTau = countTau + 1;
    relatedSigma = [relatedArdSigma(1, :); tau(countTau)*relatedArdSigma(2:end, :)];


    isThereZero = logical(uniqueVal(1) == 0);
    
    tempB = b(:, find(tree(k, :) == u1(j)), k);
    oldB = tempB(:, relatedInd(uniqueVal ~= 0));
    
    nClass = relatedNClass(k).a(j).b;
    

    newB = getBetaTree(relatedD(k).a(j).b, relatedR(k).a(j).b, oldB, relatedMu, relatedSigma, e);
    

    if isThereZero
        temp = [zeros(dim1, 1), newB];
        b(:, find(tree(k, :) == u1(j)), k) = temp(:, mapInd);
    else
        b(:, find(tree(k, :) == u1(j)), k) = newB(:, mapInd);
    end    
    
    etaT = test*newB;
    temp = exp(etaT - repmat(log(sum(exp(etaT), 2)), [1, nClass, 1]));
    
    probTest(:, find(tree(k, :) == u1(j)), k) = temp(:, mapInd);    

    
% For the first 10 iterations, I do not update hyperparameters, in order to
% obtain a good mix of parameters first. If we update the hyperparameters
% right away, and by chance we first sample of "a" and "b" are rejected,
% teh hyperparemters will become very small and takes a long time for them
% to converge.    
    if iter >= 10
           
        interceptBeta = newB(1, :);
        ardSigma0(1, find(tree(k, :) == u1(j)), k) = sqrt(1/ gamrnd( (abSigmaInt(1) + length(interceptBeta)/2), 1/( (1/abSigmaInt(2)) + (.5*sum (interceptBeta .^2) ) )) );
        
        scaledBeta = newB ./ relatedArdSigma;
        reshapedBeta = reshape(scaledBeta(2:end, :), 1, inputNum*nClass);
        tau(countTau) = sqrt(1/ gamrnd( (abTau(1) + length(reshapedBeta)/2), 1/( (1/abTau(2)) + (.5*sum (reshapedBeta .^2) ) )) );
    end
        
    var(countTau).b = newB / tau(countTau); 
    
    
    end

end



if iter >= 10

    mergedBeta = cat(2, var.b);
        
    for h = 2:pTrain
        relatedBeta = mergedBeta(h, :);
        ardSigma0(h, :, :) = sqrt(1/ gamrnd( (abSigma(1) + length(relatedBeta)/2), 1/( (1/abSigma(2)) + (.5*sum (relatedBeta .^2) ) )) );
    end

    
end

pProbTest = prod(probTest, 3);

% This part writes the current values of MCMC in different files
dlmwrite('treeBeta.dat', b);
dlmwrite('treeArdSigma.dat', ardSigma0);
dlmwrite('treeTau.dat', tau);
 

% Every 10 iterations, the program saves the MCMC values, if there are
% previous runs, the new values will be added to the old ones. 
if rem(iter, nSkip) == 0
counter = counter + 1;        
sampTree(prevSampSize+counter).b = b;
sampTree(prevSampSize+counter).tau = tau;
sampTree(prevSampSize+counter).ardSigma = [ardSigma0(:, 1, 1)]';

save sampTree sampTree;

end


mean(log( sum(pProbTest .* rTestDummy, 2) ));
sampAvgLogLike(iter, 1) = mean(log( sum(pProbTest .* rTestDummy, 2) ));
[maxPred, maxInd] = max((pProbTest)'); 
predClass = maxInd';
sampError(iter, 1) = mean(logical(predClass ~= rTest));
  	
% After burning in some iterations, the program uses the posterior samples 
% for prediction on the test cases.
if iter > burnIn
    countP = countP+1;
    newProbTest = (newProbTest*(countP-1) + pProbTest)/(countP);
    testProb = (newProbTest * countP + prevProbTest * prevNSamp) / (countP + prevNSamp);
    dlmwrite('treeProb.dat', testProb);
    dlmwrite('treeNSamp.dat', (countP+prevNSamp));
end


end % End of the MCMC algorithm

    timePassed = cputime -startTime
    
% At this part, the posterior predictive probability, testProb, is estimated for test
% cases. We use these probabilites to calculate avgLogProbabily and
% accuracy rate.        
	for i = 1:nTest
        	classProb(i, 1) = testProb(i, rTest(i));
	end
        avgLogProb = mean(log(classProb));
        [maxPred, maxInd] = max(testProb'); 
        predClass = maxInd';

        accRate = mean(logical(predClass == rTest));
        result = [avgLogProb, accRate];   

        dlmwrite('resTreeTest.dat', result, '-append');
        
  
% The following function, accepts the old values of markov chain for 
% parameters and return the updated values using Hamiltonian dynamics. 
% This function calls "getE" and "getG" functions to obtain the energy and 
% derivative for current values. Note that all the parameters are accepted 
% or rejected together.    
        
function updateB = getBetaTree(d, r, b, mu0, sigma0, e)
global leapFrog;

[dim1, dim2] = size(b);

oldB = b;

    E = getE(d, r, oldB, mu0, sigma0);
    g = getG(d, r, oldB, mu0, sigma0);
    p = randn(size(b));
    H = sum(sum( .5*(p.^2) )) + E ;
    newB = b; 
    newG = g; 

    for leap=1:leapFrog
        p = p - e*newG/2;
        newB = newB + e*p;
        newG = getG(d, r, newB, mu0, sigma0);
        p = p - e*newG/2;
    end

    newE = getE(d, r, newB, mu0, sigma0);
    newH = sum(sum( .5*(p.^2) ))+ newE ;

    acceptProb = min(1, exp(H - newH)); 

    if (rand < acceptProb)
        updateB = newB;
    else
        updateB = oldB;
    end

    
% This calculates the energy function for the posterior distribution. 
    
function E = getE(d, r, beta, mu, sigma)
nClass = length(r(1, :));

eta = d*beta;

m = max(eta');
logLike = sum( (sum((r.*eta), 2) - (m' + log(sum(exp(eta - repmat(m', 1, nClass) ), 2)) ) ) );
logPrior =  sum(sum( ( -0.5*(beta - mu).^2 ) ./ (sigma.^2) ));

E = -(logLike + logPrior);


% This part calculates the derivatives for all parameters. 
% Note that the calculation is completely vectorized and quite fast. 
% Moreover, the code is written in a way that avoids overflow.

function g = getG(d, r, beta, mu, sigma)
nClass = length(r(1, :));

eta = d*beta;

m = max(eta');

dLogLike = (d'*r - d'*exp( eta - repmat( (m' + log(sum(exp(eta - repmat(m', 1, nClass) ), 2)) ) , 1, nClass) ) );
dLogPrior =  -(beta - mu) ./ (sigma.^2);

g = -(dLogLike + dLogPrior);

