# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:41:18 2020

@author: frani
"""

import torch
import numpy as np

class PrepData():
    '''
    List of object attributes

    tabla: data frame
    idMat: matrix containing patient IDs
    contColNames : names of attributes with continuous values
    catgColNames : names of attributes with categorical values
    numPoints : number of data points
    maxVals : tensor containing maximum value of every continuous attribute
    contMat : matrix with continuous variables (in [0,1]U{-1})
    catgValList : catgValList[k] contains the possible values of k-th categorical attribute 
    catgValInds : catgValList[k] contains the indices of k-th categorical attribute in catgMat
    catgMat : matrix with one-hotted categorical attributes
    maskMat : matrix with 1's indicating categorical attribute is present and 0's if not
    contCinds : list of indices in contMat that correspond to conditioning attributes
    contXinds : list of indices in contMat that correspond to non-conditioning attributes
    catgCinds : list of indices in catgMat that correspond to conditioning attributes
    catgXinds : list of indices in catgMat that correspond to non-conditioning attributes 
    '''

    def __init__(self, tabla):

        self.tabla = tabla
        self.idMat = torch.Tensor(self.tabla['ID'].values).view(-1,1)


        # Classify variables as continuous/categorical
        colNames = tabla.columns.values[1:]
        self.contColNames = []
        self.catgColNames = []
        for colName in colNames:
            col = np.array(tabla[colName].values)
            maxVal = max(col)
            if maxVal < 20: # if the maximum value of an attribute is over 20, we'll say it's continuous
                count = 0
                for val in col:
                    if int(val) != val: # if values aren't integers, we'll say it's continuous
                        count += 1 
                        break
                if count == 0:
                    self.catgColNames.append(colName)
                else:
                    self.contColNames.append(colName)
            else:
                self.contColNames.append(colName)
        

        # put continuous variables in [0,1] (except NaNs, here represented as -1)
        self.numPoints = len(self.tabla[self.contColNames[0]].values)
        self.maxVals = torch.empty(1,len(self.contColNames))
        self.contMat = torch.empty(self.numPoints,len(self.contColNames))
        for k in range(len(self.contColNames)):
            col = np.array(tabla[self.contColNames[k]].values)
            self.maxVals[0,k] = max(col)
            col = torch.Tensor(col)
            self.contMat[:,k] = torch.where(col == -1., col, col/self.maxVals[0,k])


        # Make a onehotter for categoricals, with a NaN indicator
        self.catgValList = []
        self.catgValInds = []
        numOH = 0
        for k in range(len(self.catgColNames)):
            valList = [-1]
            for val in np.array(tabla[self.catgColNames[k]].values):
                if val not in valList:
                    valList.append(val)
            valList.remove(-1) # absent attribute is taken out
            self.catgValList.append(valList)
            self.catgValInds.append(list(range(numOH,numOH+len(valList))))
            numOH += len(valList)
        self.catgMat = torch.zeros(self.numPoints,numOH)
        self.maskMat = torch.ones(self.numPoints,numOH)
        currentIdx = 0
        for k in range(len(self.catgColNames)):
            col = np.array(tabla[self.catgColNames[k]].values)
            for j in range(len(col)):
                if col[j] == -1:
                    self.maskMat[j,currentIdx:currentIdx+len(self.catgValList[k])] = 0
                else:
                    for i in range(len(self.catgValList[k])):
                        if col[j] == self.catgValList[k][i]:
                            self.catgMat[j,currentIdx+i] = 1.
                            break
            currentIdx += len(self.catgValList[k])


    # Build lists of indices corresponding to chosen conditioning attributes
    def getInds(self,contCnames, catgCnames):
        self.contCinds = [k for k in range(len(self.contColNames)) if self.contColNames[k] in contCnames]
        self.contXinds = []
        for k in range(len(self.contColNames)):
            if k not in self.contCinds:
                self.contXinds.append(k)
        catgCcolInds = []
        for k in range(len(self.catgColNames)):
            if self.catgColNames[k] in catgCnames:
                catgCcolInds.append(k)
        lista = [self.catgValInds[k] for k in catgCcolInds]
        self.catgCinds = [k for listita in lista for k in listita]
        self.catgXinds = []
        for k in range(self.catgMat.size(1)):
            if k not in self.catgCinds:
                self.catgXinds.append(k)