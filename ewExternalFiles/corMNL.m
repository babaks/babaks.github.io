function corMNL(numberOfIterations, burnIn)
global tree nLeaf depth leapFrog; 
global eA eB;


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
% corMNL(N, n), where "N" is the number
% of MCMC iterations, and "n" is the number of discarded MCMC samples
% (i.e., burn-in).

% The current values of each MCMC iteration will be written in several files.
% These files are corAlpha (intercepts), corBeta.dat (regression parameters), 
% corSigmaA (variance of intercepts), corArdSigma.dat (ARD hyperparameters) 
% and corTau.dat (scale parameter). Since the beta matrix is three dimensional 
% (i.e., the 3rd dimension corresponds to the levels of the tree), MATLAB 
% writes the elements of beta(:, :, 1) first, beta(:, :, 2) second and ...
% The ARD and scale hyperparameters are only updated after the 10th 
% iteration. This is to make sure that alpha's and beta's are updated first. 
% Note that these files are only used as the initial values for future runs. 
% The trace of MCMC samples are written in another file called sampCor.mat, 
% which saves the values at iterations 10, 20, 30, and so forth. The performance
% of the model, measured in terms of average log-probability and accuracy, for
% test cases is wirtten in a file called "resCorTest".
 

% *************************************

% ***************** Fixed parameters

abSigmaInt = [.5, 1]; % Intercept ~ Gamma(.5, 1)
abSigma = [1, 10]; % ARD parameter: \rho ~ Gamma(1, 10)
abTau = [.5, 100]; % Scale parameter" \tau ~ Gamma(.5, 20)

leapFrog = 500; % Number of leap-frogs in Hamiltonian dynamics
eA = 0.02; % Stepsize, \epsilon, for intercepts
eB  = 0.01; % Stepsize, \epsilon, for other parameters

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

% I add a column of 1's to the begining of the test set. This is used as
% the multiplier for the intercept parameter.
test  = [ones(nTest, 1),  test ];

% For simplicity, I use "d" and "r" for training data and training
% response variable.
d = train;
r = rTrainDummy;

% As mentioned above, depth and nLeaf are the number of levels and the
% number of end nodes in the tree structure.
[depth, nLeaf] = size(tree);

% *************************************

% ***************** Initial values

% These are the initial values, if there are already inital values from the
% last runs, which are stored in the same directory, these values will be
% ignored.

a = zeros(1, nLeaf); % the intercept parameter
b = zeros(inputNum, nLeaf, depth); % These corrspond to
% the \phi parameters in our paper. As we can see, they are
% organized in a matrix of p*nLeaf*depth where "p" is the number of
% covariates, "nLeaf" is the same as the number of classes and
% "depth" is the number of tree levels. Summing these parameters
% over the third dimension gives the usual regression parameters
% for MNL.

% These two are the mean of the intercepts and \phi's. Here, mu0A and mu0 
% are fixed and will not be updated.
mu0A = zeros(1, nLeaf); % The mean of intercept parameters
mu0 = zeros(inputNum, nLeaf, depth); % The mean of other parameters

sigma0A = ones(1, nLeaf); % The hyperparameter for intercepts

% The variance of regression parameters, \phi, is the prduct of the
% following two hyperparameters.
ardSigma0 = ones(inputNum, nLeaf, depth); % The ARD hyperparameters  
tau = ones(1, 100); % The scale paramter, 100 is a number which is larger than
% number of nodes. At the begining, I use a biger vector to store tau's
% since the number of nodes is not calculated yet. I will calculated the
% actual number of nodes later.

% The number of previous MCMC samples and the corrsponding probability
% distribution. If there are previous samples, these will be changed later.
prevNSamp = 0;
prevProbTest = 0;


% This part checks whether there are previous samples available which can
% be used as starting point. These are used for running MCMC algorithms in
% more than one session.
msgId = '';
try 
   readBeta = dlmread('corBeta.dat');
   readArdSigma = dlmread('corArdSigma.dat');
catch
   [errmsg, msgId] = lasterr;
   msgId = lower(msgId);
end


% If there are initial values, this part of the code reads them in.
if isempty(msgId)
    a = dlmread('corAlpha.dat');
    b = reshape(dlmread('corBeta.dat'), [inputNum, nLeaf, depth]);
    sigma0A = dlmread('corSigmaA.dat');
    ardSigma0 = reshape(dlmread('corArdSigma.dat'), [inputNum, nLeaf, depth]);
    tau = dlmread('corTau.dat');
end

prevSampSize = 0;
msgId = '';
try 
    load sampCor;
catch
    [errmsg, msgId] = lasterr;
    msgId = lower(msgId);
    if msgId ~= 'matlab:load:couldnotreadfile'
        load sampCor;
        prevSampSize = length(sampCor);
        prevProbTest = dlmread('corProb.dat');    
        prevNSamp = dlmread('corNSamp.dat');
    end
end


% bBar corresponds to the regression parameters in MNL.
bBar = sum(b, 3);

% This is another tree created based on the orginal tree and is used for
% simplifying the book-keeping of hyperparameters.
otherTree = [ones(1, nLeaf); tree; 1:nLeaf];                                          


randn('state', sum(100*clock));
newProbTest = zeros(nTest, nLeaf);
countP = 0;
counter = 0;
startTime = cputime;

% *************************************

% ***************** MCMC algorithm

for iter = 1:numberOfIterations
iter
% This part identifies the parameters associated with each node.
countTau = 0;
for k = 1:depth
    u1 = unique(otherTree(k, :));
        
    for j = 1:length(u1)

        if u1(j) == 0
            sigma0(:, find(otherTree(k, :) == u1(j)), k) = 1;
            continue;
        end

        [uniqueVal, relatedInd, mapInd] = unique(otherTree(k+1, otherTree(k, :) == u1(j)));

        if length(uniqueVal) == 1
            sigma0(:, find(otherTree(k, :) == u1(j)), k) = 1;
            continue;
        end        
 
        isThereZero = logical(uniqueVal(1) == 0);
    
        tempArdSigma = ardSigma0(:, find(otherTree(k, :) == u1(j)), k);
        relatedArdSigma = tempArdSigma(:, relatedInd(uniqueVal ~= 0));    
        countTau = countTau + 1;
        tempSigma0 = tau(countTau)*relatedArdSigma;
        if isThereZero
            temp = [ones(size(tempSigma0(:, 1))), tempSigma0];
            sigma0(:, find(otherTree(k, :) == u1(j)), k) = temp(:, mapInd);
        else
            sigma0(:, find(otherTree(k, :) == u1(j)), k) = tempSigma0(:, mapInd);
        end
        
    end
end

% Call function getBetaCor to obtain the updated values for parameters
[a, b] = getBetaCor(d, r, a, b, mu0A, sigma0A, mu0, sigma0);

% Get the regression parameter by summing over b's.
bBar = sum(b, 3);

% For the first 10 iterations, I do not update hyperparameters, in order to
% obtain a good mix of parameters first. If we update the hyperparameters
% right away, and by chance we first sample of "a" and "b" are rejected,
% teh hyperparemters will become very small and takes a long time for them
% to converge.
if iter >= 10
    
% At this part, given the updated values of paramters, new values for 
% hyperparameters will be sampled using Gibbs sampling.

countTau = 0;
for k = 1:depth
    u1 = unique(otherTree(k, :));
        
    for j = 1:length(u1)

        if u1(j) == 0
            continue;
        end

        [uniqueVal, relatedInd, mapInd] = unique(otherTree(k+1, otherTree(k, :) == u1(j)));

        if length(uniqueVal) == 1
            continue;
        end        
 
        isThereZero = logical(uniqueVal(1) == 0);
        
        tempB = b(:, find(otherTree(k, :) == u1(j)), k);
        relatedBeta = tempB(:, relatedInd(uniqueVal ~= 0));
        
        tempArdSigma = ardSigma0(:, find(otherTree(k, :) == u1(j)), k);
        relatedArdSigma = tempArdSigma(:, relatedInd(uniqueVal ~= 0));   

        countTau = countTau+1;
        scaledBeta = relatedBeta ./ relatedArdSigma;
        reshapedBeta = reshape(scaledBeta, 1, inputNum*length(scaledBeta(1, :) ));
        tau(countTau) = sqrt(1/ gamrnd( (abTau(1) + length(reshapedBeta)/2), 1/( (1/abTau(2)) + (.5*sum (reshapedBeta .^2) ) )) );
        
        var(countTau).b = relatedBeta / tau(countTau); 

    end    
end   
   
    sigma0A(1, :) = sqrt(1/ gamrnd( (abSigmaInt(1) + length(a)/2), 1/( (1/abSigmaInt(2)) + (.5*sum (a .^2) ) )) );


    mergedBeta = cat(2, var.b);
        
    for h = 1:inputNum
        relatedBeta = mergedBeta(h, :);
        ardSigma0(h, :, :) = sqrt(1/ gamrnd( (abSigma(1) + length(relatedBeta)/2), 1/( (1/abSigma(2)) + (.5*sum (relatedBeta .^2) ) )) );
    end

end
 

% This part writes the current values of MCMC in different files
dlmwrite('corAlpha.dat', a);
dlmwrite('corBeta.dat', b);
dlmwrite('corSigmaA.dat', sigma0A);
dlmwrite('corArdSigma.dat', ardSigma0);
dlmwrite('corTau.dat', tau);
 
% Every 10 iterations, the program saves the MCMC values, if there are
% previous runs, the new values will be added to the old ones. 
if rem(iter, nSkip) == 0
  counter = counter + 1;
  sampCor(prevSampSize+counter).a = a;
  sampCor(prevSampSize+counter).b = b;
  sampCor(prevSampSize+counter).bBar = bBar;  
  sampCor(prevSampSize+counter).tau = tau;
  sampCor(prevSampSize+counter).sigma0A = sigma0A;  
  sampCor(prevSampSize+counter).ardSigma = [ardSigma0(:, 1, 1)]';
  save sampCor sampCor;
end


% After burning in some iterations, the program uses the posterior samples 
% for prediction on the test cases.
if iter > burnIn
    countP = countP+1;                
    etaT = test*[a; bBar];    
    pProbTest = exp(etaT - repmat(log(sum(exp(etaT), 2)), [1, nLeaf, 1]));   
    
    newProbTest = (newProbTest*(countP-1) + pProbTest)/(countP);
    testProb = (newProbTest * countP + prevProbTest * prevNSamp) / (countP + prevNSamp);
    dlmwrite('corProb.dat', testProb);
    dlmwrite('corNSamp.dat', (countP+prevNSamp));
end

  
end % End of the MCMC algorithm

    timePassed = cputime - startTime

% At this part, the posterior predictive probability, testProb, is estimated for test
% cases. We use these probabilites to calculate avgLogProbabily and
% accuracy rate.
    classProb = sum(testProb.*rTestDummy, 2);
    avgLogProb = mean(log(classProb));

    [maxPred, maxInd] = max(testProb'); 
    predClass = maxInd';
    accRate = mean(logical(predClass == rTest));
    result = [avgLogProb, accRate];   
    
    dlmwrite('resCorTest.dat', result, '-append');


% The following function, accepts the old values of markov chain for 
% parameters and return the updated values using Hamiltonian dynamics. 
% This function calls "getE" and "getG" functions to obtain the energy and 
% derivative for current values. Note that all the parameters are accepted 
% or rejected together.    
    
function [updateA, updateB] = getBetaCor(d, r, a, b, mu0A, sigma0A, mu0B, sigma0B)
global tree eA eB logTree depth; 
global leapFrog;

[dim1, dim2, dim3] = size(b);
bBar = sum(b, 3);
    
oldA = a;
oldB = b;

EoldLike = getE(d, r, a, bBar);

gOldLikeA = zeros(size(a));
gOldPriorA = zeros(size(a));
gOldLikeB = zeros(size(b));
gOldPriorB = zeros(size(b));

[gOldLikeA, gTemp1B] = getG(d, r, a, bBar);

pA = zeros(size(a));
pB = zeros(size(b));

logPriorA = sum( ( -0.5*(a - mu0A).^2 ) ./ (sigma0A.^2) );
EoldPrior = -logPriorA;

pA = randn(size(a));
oldPotentialA = sum( .5*(pA.^2) );
oldPotential = oldPotentialA;

dLogPriorA =  -(a - mu0A) ./ (sigma0A.^2);
gOldPriorA = -dLogPriorA;

for k = 1:dim3
    
    [uniqueVal, relatedInd, mapInd] = unique(tree(k, :));
    isThereZero = logical(uniqueVal(1) == 0);
    
    relatedMu = mu0B(:, relatedInd(uniqueVal~=0), k);
    relatedSigma = sigma0B(:, relatedInd(uniqueVal~=0), k);
    relatedBeta = b(:, relatedInd(uniqueVal ~= 0), k);
    
    logPrior = sum(sum( ( -0.5*(relatedBeta - relatedMu).^2 ) ./ (relatedSigma.^2) ));
    EoldPrior = EoldPrior + (-logPrior);  
    
    dLogPrior =  -(relatedBeta - relatedMu) ./ (relatedSigma.^2);
    gTempPrior = -dLogPrior;
    
        
    if isThereZero
        dU(k).dB = dummyvar(tree(k, :) + 1);
        dU(k).dB = dU(k).dB(:, 2:end);
        gTemp2 = [zeros(dim1, 1), gTemp1B * dU(k).dB];
        gTempPrior = [zeros(dim1, 1), gTempPrior];
        newPTemp = randn(dim1, length(find(uniqueVal~=0)));
        pTemp = [zeros(dim1, 1), newPTemp];
        oldPotential = oldPotential + sum(sum( .5*(newPTemp.^2) ));
    else
        dU(k).dB = dummyvar(tree(k, :));
        gTemp2 = gTemp1B * dU(k).dB;        
        pTemp = randn(dim1, length(uniqueVal));
        oldPotential = oldPotential + sum(sum( .5*(pTemp.^2) ));
    end
    
    gOldLikeB(:, :, k) = gTemp2(:, mapInd);
    gOldPriorB(:, :, k) = gTempPrior(:, mapInd);
    
    pB(:, :, k) = pTemp(:, mapInd);

end

    E = EoldLike + EoldPrior;
    gA = gOldLikeA + gOldPriorA;
    gB = gOldLikeB + gOldPriorB;
    
    H = oldPotential + E ;
    
    newA = oldA;
    newB = oldB;
    newGA= gA;
    newGB = gB; 
    
    gNewLikaA = zeros(size(a));
    gNewLikeB = zeros(size(b));
    gNewPriorA = zeros(size(a));
    gNewPriorB = zeros(size(b));
    
    for leap=1:leapFrog
        pA = pA - eA * newGA/2;
        pB = pB - eB * newGB/2;
        
        newA = newA + eA * pA;
        newB = newB + eB * pB;
        newBBar = sum(newB, 3);
        
        [gNewLikeA, gTemp1B] = getG(d, r, newA, newBBar);
        dLogPriorA =  -(newA - mu0A) ./ (sigma0A.^2);
        gNewPriorA = -dLogPriorA;
        

        for k = 1:dim3
            [uniqueVal, relatedInd, mapInd] = unique(tree(k, :));
            isThereZero = logical(uniqueVal(1) == 0);

            relatedMu = mu0B(:, relatedInd(uniqueVal~=0), k);
            relatedSigma = sigma0B(:, relatedInd(uniqueVal~=0), k);
            relatedBeta = newB(:, relatedInd(uniqueVal ~= 0), k);

            dLogPrior =  -(relatedBeta - relatedMu) ./ (relatedSigma.^2);
            gTempPrior = -dLogPrior;
            
            if isThereZero
                dU(k).dB = dummyvar(tree(k, :) + 1);
                dU(k).dB = dU(k).dB(:, 2:end);
                gTemp2 = [zeros(dim1, 1), gTemp1B * dU(k).dB];
                gTempPrior = [zeros(dim1, 1), gTempPrior];
            else
                dU(k).dB = dummyvar(tree(k, :));
                gTemp2 = gTemp1B * dU(k).dB;
            end
            
            gNewLikeB(:, :, k) = gTemp2(:, mapInd);
            gNewPriorB(:, :, k) = gTempPrior(:, mapInd);
            
        end
        
        if isfinite(sum(gNewLikeA+gNewPriorA)) & isfinite(sum(sum(sum(gNewLikeB+gNewPriorB))))
            newGA = gNewLikeA + gNewPriorA;
            newGB = gNewLikeB + gNewPriorB;
        end
        pA = pA - eA * newGA/2;
        pB = pB - eB * newGB/2;
    end

    EnewLike = getE(d, r, newA, newBBar);
    
    logPriorA = sum( ( -0.5*(newA - mu0A).^2 ) ./ (sigma0A.^2) );
    EnewPrior = -logPriorA;
    newPotentialA = sum( .5*(pA .^2) );
    newPotential = newPotentialA;
    
    
    for k = 1:dim3
        
        [uniqueVal, relatedInd, mapInd] = unique(tree(k, :));
        isThereZero = logical(uniqueVal(1) == 0);

        relatedMu = mu0B(:, relatedInd(uniqueVal~=0), k);
        relatedSigma = sigma0B(:, relatedInd(uniqueVal~=0), k);
        relatedBeta = newB(:, relatedInd(uniqueVal ~= 0), k);
        relatedP = pB(:, relatedInd(uniqueVal ~= 0), k);

        logPrior = sum(sum( ( -0.5*(relatedBeta - relatedMu).^2 ) ./ (relatedSigma.^2) ));
        EnewPrior = EnewPrior + (-logPrior);  
        newPotential = newPotential + sum(sum( .5*(relatedP .^ 2) ));
    end
        
    newE = EnewLike + EnewPrior;
    
    newH = newPotential + newE ;

    if isfinite(newE)
        acceptProb = min(1, exp(H - newH)); 
    else 
        acceptProb = 0;
    end

    if (rand < acceptProb)
        updateA = newA;
        updateB = newB;
    else
        updateA = oldA;
        updateB = oldB;
    end
    

% This calculates the energy function for the likelihood part of posterior 
% distribution. Energy function for the prior is calculated in the previous
% function, "getBetaCor".

function E = getE(d, r, alpha, beta)
global nLeaf;

[nTrain, inputNum] = size(d);
eta = repmat(alpha, nTrain, 1) + d*beta;

m = max(eta');
logLike = sum( (sum((r.*eta), 2) - (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) ) );

E = -logLike;


% This part calculates the derivatives for all parameters. As before, this
% only provides the derivative of likelihood function. The derivative of
% prior is calculated above in "getBetaCor" function. Note that the
% calculation is completely vectorized and quite fast. Moreover, the code
% is written in a way that avoids overflow.

function [gA, gB] = getG(d, r, alpha, beta)
global nLeaf;

[nTrain, inputNum] = size(d);
eta = repmat(alpha, nTrain, 1) + d*beta;

m = max(eta');

dLogLikeA = (ones(1, nTrain)*r - ones(1, nTrain)*exp( eta - repmat( (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) , 1, nLeaf) ) );

dLogLikeB = (d'*r - d'*exp( eta - repmat( (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) , 1, nLeaf) ) );

gA = -dLogLikeA;

gB = -dLogLikeB;



