# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:44:36 2019

@author: andrew
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:44:41 2017

@author: andrew

Use this to generate the strings spinflips,parity,q_j values.
"""
#    """This last version uses Takahashi and Suzuki's original paper:
#        Progress of Theoretical Physics, Vol. 48, No. 6B, December 1972
#        
#        For the string values. The main difference being that m1(=nu1) is the only exceptional m, unlike
#        the values that appear in the textbook these seem to work properly for anisotropies with
#        longer continued fractions"""

"""
m/l = (v1 l + r)/l = v1 + r/l = (v1 + 1/(v2 r + r2)/r) = (v1 + 1/(v2 + 1/(r/r2) )) etc.
"""

import numpy as np
#stringBuilder.niceStrings(stringBuilder,3,206)

class stringBuilder:
    gammaTop = None
    gammaBottom = None
    stringLength=None
    stringParity=None
    stringSigma=None
    intParam = None #to contain integers, v, m, y, p

    def printconfig(self):
        print("Top:", self.gammaTop)
        print("Bottom:", self.gammaBottom)    
        print("With p-values:")
        print(self.intParam['pMat'])
    
    def _init_(self,top,btm,params):
        self.gammaBottom=btm
        self.gammaTop=top
        if 'vMat' not in params.keys():
            params['vMat']=self.computeV(self)
        if 'yMat' not in params.keys():
            params['yMat']=self.yValues(params['vMat'])
        if 'mMat' not in params.keys():
            params['mMat']=self.mValues(params['vMat'])
        if 'pMat' not in params.keys():
            params['pMat']=self.pValues(self,params['vMat'])
        self.intParam=params
        self.printconfig(self)
    
    def computeV(self):
        v=list([self.gammaBottom//self.gammaTop])
        ell = self.gammaBottom-v[0]*self.gammaTop
        r = self.gammaTop
        i=1
        while ell != 0:
            v=v+list([r//ell]) #integer division to find term
            ellTemp=ell
            ell = r-v[i]*ell
            r = ellTemp
            i=i+1
        return(np.asarray(v))
    def yValues(v):
        y = list([0,1])
        ell=len(v)
        for j in range(2,ell+2):
            yValTemp=y[j-2]+(v[j-2])*y[j-1]
            y=y+list([yValTemp]) 
        return(np.asarray(y))
    def mValues(v):#working, outputs beginning at m0 = 0
#        v=self.vMat
        m = list([0])
        ell=len(v)
        for j in range(0,ell):
            mTempValue=0
            for i in range(0,j+1):
                mTempValue+=v[i]
            m=m+list([mTempValue]) #integer division to find term
        return(np.asarray(m))
    def pValues(self,v):
        p = list([float(self.gammaBottom)/float(self.gammaTop),1])
        for j in range(2,len(v)+2):
            tempPVal=p[j-2]-float(v[j-2])*p[j-1]
            p+=list([tempPVal]) #integer division to find term
        return(np.asarray(p))
    
    def spinflipsFull(self,gammaTop,gammaBottom):
        m,y,p=self.intParam['mMat'],self.intParam['yMat'],self.intParam['pMat']
        spins=list()
        parity=list()
        auxInt=list([])
        for i in range(0,len(m)-1):      
            for j in range(m[i],m[i+1]):
                if j==m[1]:
    #                spinTempCalc=y[i]+(j-m[i])*y[i+1]    
                    parity+=list([(-1)**(i)])
    #               spins=spins+list([spinTempCalc])
                    spins+=list([y[i]])
                    auxTempCalc=(p[i]-(j-m[i])*p[i+1])*(-1)**(i) 
                    auxInt+=list([auxTempCalc])
                else:
                    spinTempCalc=y[i]+(j-m[i])*y[i+1]
                    parity+=list([(-1)**np.abs(((spinTempCalc-1)*gammaTop)//(gammaBottom))])
    #               spins=spins+list([spinTempCalc])
                    spins+=list([spinTempCalc])
                    auxTempCalc=(p[i]-(j-m[i])*p[i+1])*(-1)**(i) 
                    auxInt+=list([auxTempCalc])                 
        spins+=list([y[len(m)-1]])
         #special case needed if top =1 then m_1 == j comes in the final step!
        if (m[i+1])==m[1]:
            parity+=list([(-1)**(np.floor(np.abs(y[len(m)-1]-1)*gammaTop/gammaBottom)-1)])
        else:
            parity+=list([(-1)**(np.floor(np.abs(y[len(m)-1]-1)*gammaTop/gammaBottom))])
        auxInt+=list([-auxInt[m[len(m)-1]-1]])            
        return(np.asarray(spins),np.asarray(parity),np.asarray(auxInt))
    
    def niceStrings(self,top,bottom):#outputs (stringLength, stringParity, sigma=sign(q))
        self._init_(self,top,bottom,dict())
        
        self.gammaTop=top
        self.gammaBottom=bottom
        stringSet=self.spinflipsFull(self,top,bottom)
    #    stringVal=np.zeros((len(stringSet[1])-1),dtype='int8,int8,float64')
        stringVal=np.zeros((len(stringSet[1])-1,3),int)
        for i in range(1,len(stringSet[1])):
            stringVal[i-1]=(int(stringSet[0][i]),int(stringSet[1][i]),np.sign(stringSet[2][i]))
        return(np.asarray(stringVal))
    


