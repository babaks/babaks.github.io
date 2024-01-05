function MNL(numberOfIterations, burnIn)
global leapFrog e nLeaf; 

% ***************** General Instruction

% To run this program, you first need to create a matlab dataset called 
% "data.mat" and put it in the same folder as this file. The dataset should
% include the following matrices: "train", "test", "rTrain" and "rTest"
% which are the training set, the test set, the response values for the
% training cases and the response values for the test cases respectively.
% "train" and "test" contain the explanatory variables. Later, a vector of 
% 1's will be added to these datasets to account for the intercept. 
% Response values are class labels, which start from 1 and increment by 1.
% To run this program, enter the command MNL(N, n), where "N" is the number
% of MCMC iterations, and "n" is the number of discarded MCMC samples 
% (i.e., burn-in). 

% The current values of each MCMC iteration will be written in several files.
% These files are mnBeta.dat (regression parameters with intercepts in the 
% first row), mnArdSigma.dat (ARD hyperparameters) and mnTau.dat (scale 
% hyperparameter). The ARD and scale hyperparameters are only updated after
% the 10th iteration. This is to make sure that beta's are updated first. 
% Note that these files are only used as the initial values for future runs. 
% The trace of MCMC samples are written in another file called sampMn.mat, 
% which saves the values at iterations 10, 20, 30, and so forth. The performance
% of the model, measured in terms of average log-probability and accuracy, for 
% test cases is wirtten in a file called "resMnTest". 
 


% ***************** Fixed parameters

abSigmaInt = [.5, 1]; % Intercept ~ Gamma(.5, 1)
abSigma = [1, 10]; % ARD parameter: \rho ~ Gamma(1, 10)
abTau = [.5, 20]; % Scale parameter" \tau ~ Gamma(.5, 20)

leapFrog = 500; % Number of leap-frogs in Hamiltonian dynamics
e = 0.02; % Stepsize, \epsilon, in Hamiltonian dynamics

nSkip = 1; % Number of MCMC samples skipped before saving the current
% values. This is just for evaluating convergances. Prediction is based
% on all samples after the initial discard.


% *************************************

% Here, I load the NIPS dataset, which is stroed in the file data.mat.
load data;

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
[nData, nLeaf] = size(allRDummy);

% I add a column of 1's to the begining of each dataset. This is used as
% the multiplier for the intercept parameter.
train = [ones(nTrain, 1), train];
test  = [ones(nTest, 1),  test ];

% For simplicity, I use "d" and "r" for training data and training
% response variable.
d = train;
r = rTrainDummy;

% *************************************

% ***************** Initial values

% These are the initial values, if there are already inital values from the
% last runs, which are stored in the same directory, these values will be
% ignored.

% Number of regression parameters including the intercept.
pTrain  = inputNum  + 1;

% The first row of regression parameters, b, is for the intercept.
b = zeros(pTrain, nLeaf);

% This is the mean of regression parameters. Here, mu0 is
% fixed and will not be updated.
mu0 = zeros(pTrain, nLeaf);

% The variance of regression parameters (except for the intercept parameter)
% is the prduct of the following two hyperparameters.
ardSigma0 = ones(pTrain, nLeaf); 
tau = 1;

% The number of previous MCMC samples and the corrsponding probability
% distribution. If there are previous samples, these will be changed later.
prevProbTest = 0;
prevNSamp = 0;


% This part checks whether there are previous samples available which can
% be used as starting point. These are used for running MCMC algorithms in
% more than one session.
msgId = '';
try 
    readBeta = dlmread('mnBeta.dat');
    readArdSigma = dlmread('mnArdSigma.dat');
catch
    [errmsg, msgId] = lasterr;
    msgId = lower(msgId);
end

% If there are initial values, this part of the code reads them in.
if isempty(msgId)
    b = dlmread('mnBeta.dat');
    ardSigma0 = dlmread('mnArdSigma.dat');
    tau = dlmread('mnTau.dat');
end

prevSampSize = 0;
msgId = '';
try 
    load sampMn;
catch
    [errmsg, msgId] = lasterr;
    msgId = lower(msgId);
    if msgId ~= 'matlab:load:couldnotreadfile'
        load sampMn;
        prevSampSize = length(sampMn);
        prevProbTest = dlmread('mnProb.dat');
        prevNSamp = dlmread('mnNSamp.dat');
    end
end

% This is the variance of regression parameters. The first one is for the
% intercept and is updated differnt from the other parameters.
sigma0 = [ardSigma0(1, :); tau*ardSigma0(2:end, :)];


randn('state', sum(100*clock));
newProbTest = zeros(nTest, nLeaf);
countP = 0;
counter = 0;
startTime = cputime;

% *************************************

% ***************** MCMC algorithm

for iter = 1:numberOfIterations
iter    
% Call function getBetaMn to obtain the updated values for parameters    
newB = getBetaMN(d, r, b, mu0, sigma0);

b = newB;


% For the first 10 iterations, I do not update hyperparameters, in order to
% obtain a good mix of parameters first. If we update the hyperparameters
% right away, and by chance we first sample of "a" and "b" are rejected,
% teh hyperparemters will become very small and takes a long time for them
% to converge.
if iter >= 10

% At this part, given the updated values of paramters, new values for 
% hyperparameters will be sampled using Gibbs sampling.

tempSigma = [];
interceptBeta = b(1, :);
tempSigma(1, 1) = sqrt(1/ gamrnd( (abSigmaInt(1) + length(interceptBeta)/2), 1/( (1/abSigmaInt(2)) + (.5*sum (interceptBeta .^2) ) )) );
  
    scaledBeta = b / tau;

    for j = 2:pTrain
        
        relatedBeta = scaledBeta(j, :);
        tempSigma(j, 1) = sqrt(1/ gamrnd( (abSigma(1) + length(relatedBeta)/2), 1/( (1/abSigma(2)) + (.5*sum (relatedBeta .^2) ) )) );
    end
    
    ardSigma0 = repmat(tempSigma, 1, nLeaf);
    
    scaledBeta = b(2:end, :) ./ ardSigma0(2:end, :);
    reshapedBeta = reshape(scaledBeta, 1, inputNum*nLeaf);

    tau = sqrt(1/ gamrnd( (abTau(1) + length(reshapedBeta)/2), 1/( (1/abTau(2)) + (.5*sum (reshapedBeta .^2) ) )) );

end

sigma0 = [ardSigma0(1, :); tau*ardSigma0(2:end, :)];

% This part writes the current values of MCMC in different files
dlmwrite('mnBeta.dat', b);
dlmwrite('mnArdSigma.dat', ardSigma0);
dlmwrite('mnTau.dat', tau);
 
% Every 10 iterations, the program saves the MCMC values, if there are
% previous runs, the new values will be added to the old ones. 
if rem(iter, nSkip) == 0
    counter = counter + 1;   
    sampMn(prevSampSize+counter).b = b;
    sampMn(prevSampSize+counter).tau = tau;
    sampMn(prevSampSize+counter).ardSigma = ardSigma0(:, 1);
    save sampMn sampMn;
end

% After burning in some iterations, the program uses the posterior samples 
% for prediction on the test cases.
if iter > burnIn
    countP = countP+1;                
    etaT = test*b;    
    pProbTest = exp(etaT - repmat(log(sum(exp(etaT), 2)), [1, nLeaf, 1]));   
    
    newProbTest = (newProbTest*(countP-1) + pProbTest)/(countP);
    testProb = (newProbTest * countP + prevProbTest * prevNSamp) / (countP + prevNSamp);
    dlmwrite('mnProb.dat', testProb);
    dlmwrite('mnNSamp.dat', (countP+prevNSamp));
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
    
    dlmwrite('resMnTest.dat', result, '-append');



% The following function, accepts the old values of markov chain for 
% parameters and return the updated values using Hamiltonian dynamics. 
% This function calls "getE" and "getG" functions to obtain the energy and 
% derivative for current values. Note that all the parameters are accepted 
% or rejected together.    

function updateB = getBetaMN(d, r, b, mu0, sigma0)
global leapFrog e nLeaf; 

[dim1, dim2] = size(b);

oldB = b;

E = getE(d, r, oldB, mu0, sigma0);
g = getG(d, r, oldB, mu0, sigma0);
p = randn(size(b));
H = sum(sum( .5*(p.^2) ))+ E;
newB = b; 
newG = g; 

for leap=1:leapFrog
    p = p - e*newG/2;
    newB = newB + e*p;
    tempG = getG(d, r, newB, mu0, sigma0);
    if isfinite(tempG)
        newG = tempG;
    end

    p = p - e*newG/2;
end

newE = getE(d, r, newB, mu0, sigma0);
newH = sum(sum( .5*(p.^2) )) + newE;

acceptProb = min(1, exp(H - newH)); 

if (rand < acceptProb)
    updateB = newB;
else
    updateB = oldB;
end


% This calculates the energy function for the posterior distribution. 

function E = getE(d, r, beta, mu, sigma)
global nLeaf;

eta = d*beta;

m = max(eta');
logLike = sum( (sum((r.*eta), 2) - (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) ) );

logPrior =  sum(sum( ( -0.5*(beta - mu).^2 ) ./ (sigma.^2) ));

E = -(logLike + logPrior);


% This part calculates the derivatives for all parameters. 
% Note that the calculation is completely vectorized and quite fast. 
% Moreover, the code is written in a way that avoids overflow.

function g = getG(d, r, beta, mu, sigma)
global nLeaf;
eta = d*beta;

m = max(eta');

dLogLike = (d'*r - d'*exp( eta - repmat( (m' + log(sum(exp(eta - repmat(m', 1, nLeaf) ), 2)) ) , 1, nLeaf) ) );
dLogPrior =  -(beta - mu) ./ (sigma.^2);
 
g = -(dLogLike + dLogPrior);

