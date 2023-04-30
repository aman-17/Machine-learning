"""
@Name: Utility.py
@Creation Date: October 3, 2022
@author: Ming-Long Lam, Ph.D.
@organization: Illinois Institute of Technology
(C) All Rights Reserved.
"""

import numpy
import pandas
import statsmodels.api as smodel

def SWEEPOperator (pDim, inputM, origDiag, sweepCol = None, tol = 1e-7):
    ''' Implement the SWEEP operator

    Parameter
    ---------
    pDim: dimension of matrix inputM, integer greater than one
    inputM: a square and symmetric matrix, numpy array
    origDiag: the original diagonal elements before any SWEEPing
    sweepCol: a list of columns numbers to SWEEP
    tol: singularity tolerance, positive real

    Return
    ------
    A: negative of a generalized inverse of input matrix
    aliasParam: a list of aliased rows/columns in input matrix
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    if (sweepCol is None):
        sweepCol = range(pDim)

    aliasParam = []
    nonAliasParam = []

    A = numpy.copy(inputM)
    ANext = numpy.zeros((pDim,pDim))

    for k in sweepCol:
        Akk = A[k,k]
        pivot = tol * abs(origDiag[k])
        if (not numpy.isinf(Akk) and abs(Akk) >= pivot):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / abs(Akk)
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = numpy.zeros(pDim)
            ANext[k, :] = numpy.zeros(pDim)
        A = ANext
    return (A, aliasParam, nonAliasParam)

def create_interaction (df1, df2):
    ''' Return the columnwise product of two dataframes (must have same number of rows)

    Parameter
    ---------
    df1: first input data frame
    df2: second input data frame

    Return
    ------
    outDF: the columnwise product of two dataframes
    '''

    name1 = df1.columns
    name2 = df2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        u = df1[col1]
        outName = col1 + " * "
        for col2 in name2:
            outDF[outName+col2] = u * df2[col2]
    return(outDF)

def binary_model_metric (target, valueEvent, valueNonEvent, predProbEvent, eventProbThreshold = 0.5):
    '''Calculate metrics for a binary classification model

    Parameter
    ---------
    target: Panda Series that contains values of target variable
    valueEvent: Formatted value of target variable that indicates an event
    valueNonEvent: Formatted value of target variable that indicates a non-event
    predProbEvent: Panda Series that contains predicted probability that the event will occur
    eventProbThreshold: Threshold for event probability to indicate a success

    Return
    ------
    outSeries: Pandas Series that contain the following statistics
               ASE: Average Squared Error
               RASE: Root Average Squared Error
               MCE: Misclassification Rate
               AUC: Area Under Curve
    '''

    # Number of observations
    nObs = len(target)

    # Aggregate observations by the target values and the predicted probabilities
    aggrProb = pandas.crosstab(predProbEvent, target, dropna = True)

    # Calculate the root average square error
    ase = (numpy.sum(aggrProb[valueEvent] * (1.0 - aggrProb.index)**2) + 
           numpy.sum(aggrProb[valueNonEvent] * (0.0 - aggrProb.index)**2)) / nObs
    if (ase > 0.0):
        rase = numpy.sqrt(ase)
    else:
        rase = 0.0
    
    # Calculate the misclassification error rate
    nFP = numpy.sum(aggrProb[valueNonEvent].iloc[aggrProb.index >= eventProbThreshold])
    nFN = numpy.sum(aggrProb[valueEvent].iloc[aggrProb.index < eventProbThreshold])
    mce = (nFP + nFN) / nObs

    # Calculate the number of concordant, discordant, and tied pairs
    nConcordant = 0.0
    nDiscordant = 0.0
    nTied = 0.0

    # Loop over the predicted event probabilities from the Event column
    predEP = aggrProb.index
    eventFreq = aggrProb[valueEvent]

    for i in range(len(predEP)):
        eProb = predEP[i]
        eFreq = eventFreq.loc[eProb]
        if (eFreq > 0.0):
            nConcordant = nConcordant + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb > aggrProb.index])
            nDiscordant = nDiscordant + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb < aggrProb.index])
            nTied = nTied + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb == aggrProb.index])

    auc = 0.5 + 0.5 * (nConcordant - nDiscordant) / (nConcordant + nDiscordant + nTied)

    outSeries = pandas.Series({'ASE': ase, 'RASE': rase, 'MCE': mce, 'AUC': auc})
    return(outSeries)

def curve_coordinates (target, valueEvent, valueNonEvent, predProbEvent):
    '''Calculate coordinates of the Receiver Operating Characteristics (ROC) curve and
    the Precision Recall (PR) curve

    Classification Convention
    -------------------------
    An observation is classified as Event if the predicted event probability is
    greater than or equal to a given threshold value.

    Parameter
    ---------
    target: Panda Series that contains values of target variable
    valueEvent: Formatted value of target variable that indicates an event
    valueNonEvent: Formatted value of target variable that indicates a non-event
    predProbEvent: Panda Series that contains predicted probability that the event will occur

    Return
    ------
    outCurve: Pandas dataframe for the curve coordinates
              Threshold: Event probability threshold of the coordinates
              Sensitivity: Sensitivity coordinate
              OneMinusSpecificity: 1 - Specificity coordinate
              Precision: Precision coordinate
              Recall: Recall coordinate
              F1Score: F1 Score
    '''

    outCurve = pandas.DataFrame()

    # Aggregate observations by the target values and the predicted probabilities
    aggrProb = pandas.crosstab(predProbEvent, target, dropna = True)

    # Find out the number of events and non-events
    n_event = numpy.sum(aggrProb[valueEvent])
    n_nonevent = numpy.sum(aggrProb[valueNonEvent])

    q00 = False
    q11 = False

    for thresh in aggrProb.index:
        nTP = numpy.sum(aggrProb[valueEvent].iloc[aggrProb.index >= thresh])
        nFP = numpy.sum(aggrProb[valueNonEvent].iloc[aggrProb.index >= thresh])

        Sensitivity = nTP / n_event
        OneMinusSpecificity = nFP / n_nonevent
        Precision = nTP / (nTP + nFP)
        Recall = Sensitivity
        F1Score = 2.0 / (1.0 / Precision + 1.0 / Recall)

        q00 = (nTP == 0.0 and nFP == 0.0)
        q11 = (nTP == n_event and nFP == n_nonevent)

        outCurve = outCurve.append([[thresh, Sensitivity, OneMinusSpecificity, Precision, Recall, F1Score]], ignore_index = True)

    if (not q00):
        outCurve = outCurve.append([[numpy.NaN, 0.0, 0.0, numpy.NaN, 0.0, numpy.NaN]], ignore_index = True)

    if (not q11):
        Sensitivity = 1.0
        OneMinusSpecificity = 1.0
        Precision = n_event / (n_event + n_nonevent)
        Recall = Sensitivity
        F1Score = 2.0 / (1.0 / Precision + 1.0 / Recall)
        outCurve = outCurve.append([[numpy.NaN, Sensitivity, OneMinusSpecificity, Precision, Recall, F1Score]], ignore_index = True)

    outCurve.columns = ['Threshold', 'Sensitivity', 'OneMinusSpecificity', 'Precision', 'Recall', 'F1Score']
    outCurve.sort_values(by = ['OneMinusSpecificity', 'Sensitivity', 'Threshold'], inplace = True)

    return (outCurve)

def LinearRegressionModel (X, y, tolSweep = 1e-7):
    ''' Train a linear regression model

    Parameter
    ---------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    A list of model output:
    (0) b: an array of regression coefficient
    (1) residual_SS: residual sum of squares
    (2) XtX_Ginv: a generalized inverse of the XtX matrix
    (3) aliasParam: a list of aliased rows/columns in input matrix
    (4) nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    # X: A Pandas DataFrame, rows are observations, columns are regressors
    # y: A Pandas Series, rows are observations of the response variable

    Z = X.join(y)
    n_sample = Z.shape[0]
    n_param = Z.shape[1] - 1

    ZtZ = Z.transpose().dot(Z)
    diag_ZtZ = numpy.diagonal(ZtZ)
    eps_double = numpy.finfo(numpy.float64).eps
    tol = numpy.sqrt(eps_double)

    ZtZ_transf, aliasParam, nonAliasParam = SWEEPOperator ((n_param+1), ZtZ, diag_ZtZ, sweepCol = range(n_param), tol = tol)

    b = ZtZ_transf[0:n_param, n_param]
    b[aliasParam] = 0.0

    XtX_Ginv = - ZtZ_transf[0:n_param, 0:n_param]
    XtX_Ginv[:, aliasParam] = 0.0
    XtX_Ginv[aliasParam, :] = 0.0

    residual_SS = ZtZ_transf[n_param, n_param]

    return ([b, residual_SS, XtX_Ginv, aliasParam, nonAliasParam])

def MNLogisticModel (X, y, maxIter = 20, tolSweep = 1e-7):
    ''' Train a Multinomial Logistic Model

    Parameter
    ---------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable
    maxIter: Maximum number of iterations
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    A list of model output:
    (0) mFit: the Fit object of MNLogit
    (1) mLLK: model log-likelihood value
    (2) mDF: model degrees of freedom
    (3) mParameter: model parameter estimates5
    (4) aliasParam: indices of aliased parameters
    (5) nonAliasParam: indices of non-aliased parameters
    '''

    n_param = X.shape[1]

    # Identify the aliased parameters
    XtX = X.transpose().dot(X)
    origDiag = numpy.diag(XtX)
    XtXGinv, aliasParam, nonAliasParam = SWEEPOperator (n_param, XtX, origDiag, sweepCol = range(n_param), tol = tolSweep)

    # Train a multinominal logistic model
    X_reduce = X.iloc[:, list(nonAliasParam)]
    mObj = smodel.MNLogit(y, X_reduce)
    mFit = mObj.fit(method = 'newton', maxiter = maxIter, tol = 1e-6, full_output = True, disp = True)
    mLLK = mFit.llf
    mDF = len(nonAliasParam) * (mFit.J - 1)
    mParameter = mFit.params

    # Return model statistics
    return ([mFit, mLLK, mDF, mParameter, aliasParam, nonAliasParam])

