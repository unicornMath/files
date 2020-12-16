# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:55:17 2019

@author: andrew
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 09:25:46 2018

@author: andrew
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 06:15:54 2018

@author: andrew
"""

"""
Created on Sun Feb  4 14:50:00 2018

@author: andrewurichuk
"""
   
import numpy as np #need this.
import matplotlib.pyplot as plt #used in the computeChargesStep
import xxz_stringBuilderClass as xxzString

"""Useful for checking the plots"""
def plotterSing(M,constants,top,bottom,valueGraph,nameList,ptType='--'):
    # constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
    rapIncNumber=(valueGraph[0,0].shape)[0]; minRap=constants[1]; stringNumber=np.int(constants[3])
    print(constants[5])
    r = np.zeros(rapIncNumber,float)
    for i in range(0,rapIncNumber):
        r[i] = (minRap + i * (-2*minRap)/(rapIncNumber))
    for j in range(0,len(valueGraph)):
        plt.figure(j+M)
        for k in np.arange(stringNumber):#feed in only string level functions in the array        
            plt.plot(r,valueGraph[j][k],ptType)      
    #Demonstrates only small shifts the fermiFactor, due to temperature differences
    # Notice that the midpoint falls roughly between the two outer data points
    #It is expected that th lines would be filled in within the system region.
        plt.xlabel('Rapidity k')
        plt.title(nameList[j] + ' for ' + r'$\gamma=\frac{' +str(top)+r'\pi}{'+str(bottom) + r'}$'+ ', beta=%4.f'%(constants[5]))
        plt.legend(['string %.0f' %i for i in np.arange(stringNumber)])
        plt.grid(True)
        plt.show()
"""
Begin definitions for kernels and scattering functions
"""

class xxzKer:#contains kernel information
    constants = None# constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
    string=None
    
    def _init_(self,consts=dict()):
        self.constants=consts
        self.constants['string']=xxzString.stringBuilder.niceStrings(xxzString.stringBuilder,consts['top'],consts['btm'])
        print(consts)
        
    def aCountFld(self,k):#using q-values eqn. 9.50 in Takahashi
        # constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
    #    alpha=constants[2]
        fld=self.constants['field']
        v_p, q = self.constants['string'].T[1], self.constants['string'].T[0]
        gma = self.constants['gamma']
        stringNum=v_p.shape[-1]; dsc = k.shape[-1]
        val1=np.reshape(v_p*gma*np.sin(gma*q)/(2*np.pi),(stringNum,1))
        zeroCheck=1-np.sign(np.abs(np.reshape(np.cosh(gma*k),(1,dsc))-np.reshape(v_p*np.cos(gma*q),(stringNum,1))))
        val2=(np.reshape(np.cosh(gma*k),(1,dsc))-np.reshape(v_p*np.cos(gma*q),(stringNum,1))+zeroCheck) #to avoid 0/0 at rapidity=0, q=0
        return(val1/val2+np.reshape(q*fld/2/np.pi,(stringNum,1)))    

    def aCountCoord(self,k):#using q-values eqn. 9.50 in Takahashi
        # constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
    #    alpha=constants[2]
        v_p, q = self.constants['string'].T[1], self.constants['string'].T[0]
        gma = self.constants['gamma']
        stringNum=v_p.shape[-1]; dsc = k.shape[-1]
        val1=np.reshape(v_p*gma*np.sin(gma*q)/(2*np.pi),(stringNum,1))
        zeroCheck=1-np.sign(np.abs(np.reshape(np.cosh(gma*k),(1,dsc))-np.reshape(v_p*np.cos(gma*q),(stringNum,1))))
        val2=(np.reshape(np.cosh(gma*k),(1,dsc))-np.reshape(v_p*np.cos(gma*q),(stringNum,1))+zeroCheck) #to avoid 0/0 at rapidity=0, q=0
        return(val1/val2)    
        
    def aCountPrimeCoord(self,k):#using q-values eqn. 9.50 in Takahashi
    # constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
#    alpha=constants[2]
        v_p, q = self.constants['string'].T[1], self.constants['string'].T[0]
        gma = self.constants['gamma']
        stringNum=v_p.shape[0]; dsc = k.shape[0]
        val1=-np.reshape(v_p*(gma**2)/(2*np.pi)*np.sin(gma*q),(stringNum,1))*np.sinh(gma*k)
        zeroCheck=1-np.sign(np.abs(np.reshape(np.cosh(gma*k),(1,dsc))-np.reshape(v_p*np.cos(gma*q),(stringNum,1))))
        val2=(np.reshape(np.cosh(gma*k),(1,dsc))-np.reshape(v_p*np.cos(gma*q),(stringNum,1))+zeroCheck) #to avoid 0/0 at rapidity=0, q=0/cosalpha1=1
        return(val1/(val2)**2)  
    def aCount_X(self,k,q,v_p):#used for the x-coordinate kernel
        # constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
    #    alpha=constants[2]
        gma = self.constants['gamma']
        val1=(v_p*gma*np.sin(gma*q)/(2*np.pi))
        zeroCheck=1-np.sign(np.abs((np.cosh(gma*k))-(v_p*np.cos(gma*q))))
        val2=((np.cosh(gma*k))-(v_p*np.cos(gma*q))+zeroCheck) #to avoid 0/0 at rapidity=0, q=0
        return(val1/val2)            
    def aCount_F(self,k,n,v_p): #use this to construct the t-krnl
#        v_p, n = self.constants['string'].T[1], self.constants['string'].T[0]
        gma=self.constants['gamma']; 
        qHalf=np.int(n/(2*np.pi/gma))//2; q=qHalf*2
    #    q=np.int(n/(2*np.pi/gma))
        if np.abs(np.sin(n*gma))<10**-12:
            tmpVal=0
        elif v_p==1:
            qHalf = (np.int(n*gma/(np.pi)))//2;q = 2*qHalf
            zeroVal=((n-(1+q)*np.pi/gma))/(np.pi/gma)
            tmpVal=-(np.sinh((n-(1+q)*np.pi/gma)*k)/(np.sinh(k*np.pi/gma)+(1-np.sign(np.abs(k)))))-zeroVal*(1-np.sign(np.abs(k)))    
        elif v_p==-1:
            qHalf = np.int(n*gma/np.pi +1)//2; q = 2*qHalf
            zeroVal=-((n-(q)*np.pi/gma))/(np.pi/gma)
            tmpVal=-(np.sinh((n-(q)*np.pi/gma)*k)/(np.sinh(k*np.pi/gma)+(1-np.sign(np.abs(k)))))+zeroVal*(1-np.sign(np.abs(k)))
        else:
            tmpVal=0
            print("weird problem, no cases satisfied")
    #    return(n,tmpVal,q)
        return(tmpVal)
    def tKrnl_F(self,str1,str2,k): #yields the XXZ kernel in one variable (to be fourierTransformed back)
# constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
        string=self.constants['string']
        tempTcount=0
        j=np.int(str1);    h=np.int(str2)
        tempTcount+=-self.aCount_F(self,k,np.abs(string[j,0]-string[h,0]),string[h,1]*string[j,1])-self.aCount_F(self,k,np.abs(string[j,0]+string[h,0]),string[h,1]*string[j,1])
        for m in np.arange(0,np.minimum(np.int(string[j,0]),np.int(string[h,0]))+1):
            tempTcount+=2*self.aCount_F(self,k,np.abs(string[j,0]-string[h,0])+2*m,string[j,1]*string[h,1])
        return(np.asarray(tempTcount))
    def tKrnl_X(self,str1,str2,k): #yields the XXZ kernel in one variable (to be fourierTransformed back)
# constants[rapIncNumber,minRap,gamma,stringNumber,betaL,betaR,mu]
        string=self.constants['string']
        tempTcount=0
        j=np.int(str1);    h=np.int(str2)
        tempTcount+=-self.aCount_X(self,k,np.abs(string[j,0]-string[h,0]),string[h,1]*string[j,1])-self.aCount_X(self,k,np.abs(string[j,0]+string[h,0]),string[h,1]*string[j,1])
        for m in np.arange(0,np.minimum(np.int(string[j,0]),np.int(string[h,0]))+1):
            tempTcount+=2*self.aCount_X(self,k,np.abs(string[j,0]-string[h,0])+2*m,string[j,1]*string[h,1])
        return(np.asarray(tempTcount))
    
#End kernel definitions
class xxzNLIE:
    constants=None
    string=None
    m_kernel=None
    m_driving=None
    d_results=None
    
    def _init_(self,consts,kernel,driving):
        self.constants=consts
        self.string=consts['string']
        self.m_kernel=kernel
        self.m_driving=driving
        self.d_results=dict()
        self.m_disc=consts['disc']

    """return fourier space thermal energy computed via DFT using Klumper Trick"""
    def yangYangE(self,itrn=10,acc=10**-14):
        beta=self.constants['beta']; gma=self.constants['gamma']
        stpNum=0;  flipString=np.moveaxis(self.string,0,1)# 1 - string length 2 - parity 3 - auxiliary integers
        e_X=-2*np.pi*(np.sin(gma)/gma)*self.m_driving; tCount_F= self.m_kernel
        sigmaMat=np.sign(flipString)[2]#auxiliary signs
        dsc=np.int(self.constants['steps']);    #dK = 2*np.abs(mnX)/discretization; 
        pmAry=np.array([np.ones(dsc//2),-np.ones(dsc//2)]).T.flatten(); #tCount_F=np.fft.fft(pmAry*tCount_X[:,:])#[i,j,K]   *np.reshape(sigmaMat,(1,stringNum,1))
    #    tCountShift_F=np.fft.ifftshift(tCount_F[:,:])#either shift or inverse shift can be used (identical for even discretizations)
        gsVal=e_X*beta*pmAry
        diff=np.abs(5); stpNum+=1; diff2=4
    #    inmt=1
        while stpNum<10000: #ensuring convergence 
            
            if np.amax(np.abs(diff))>acc:
                if np.amax(np.abs(diff))==np.amax(np.abs(diff2)):
                    stpNum+=20000
                for iterVal in range(1,itrn):
                    logPlusE_X=np.logaddexp(0,-pmAry*gsVal[:]) #shift so everything is in the center (-1)**m term accounted for by pmAry
                    logPlusE_F=np.fft.fft((pmAry*logPlusE_X)[:])
                    convMat_X=np.fft.ifft((tCount_F*logPlusE_F)[:,:])
                    convVec_X=np.tensordot(convMat_X,sigmaMat,axes=(1,0)) #[i,K]
                    tmpGs=e_X*beta*pmAry+np.real(convVec_X)#*(pmAry.reshape(1,dsc))   
                    diff2=diff
                    diff=np.abs(gsVal-tmpGs)
                    newVal=(tmpGs+gsVal)/np.float(2)# half steps seem sufficiently stable (full steps get stuck often)
                    gsVal=newVal
                    stpNum+=1
                print(np.amax(np.abs(diff)),np.amax(np.abs(diff2)),stpNum)        
            else:
                stpNum+=20000
        tempResult=gsVal*pmAry 
        self.d_results['fermi_X_t0_error']=diff
        self.d_results['fermi_X_t0']=tempResult
#        print(sigmaMat)
        return(np.asarray([tempResult,diff]))
    
    def dressChargeAdj(self,bareCharge,fermi_X,acc=10**-12):#same setup as above, could potentially be solved linearly, but works fine
        #used when fermi_X has an additional axis to the usual [str,rap] axes (for example an x-axis)
        #ie. fermi_X [xPos,string,rapidity]
        tCount_F= self.m_kernel #[str,str,K]
        stpNum=0; flipString=np.moveaxis(self.string,0,1)# 1 - string length 2 - parity 3 - auxiliary integers
        sigmaMat=np.sign(flipString)[2]#auxiliary signs
        bShape=bareCharge.shape;    dsc=bShape[-1];    #dK = 2*np.abs(mnX)/discretization; 
        pmAry=np.array([np.ones(dsc//2),-np.ones(dsc//2)]).T.flatten(); #
    #    tCountShift_F=np.fft.ifftshift(tCount_F[:,:])#either shift or inverse shift can be used (identical for even discretizations)
        gsVal=bareCharge*pmAry#[xPos,str,K]
        diff=np.abs(1); stpNum+=1; diff2=10**30
        while stpNum<10000: #ensuring convergence              
            if np.amax(np.abs(diff))>acc:
                for iterVal in np.arange(10):
                    mixMat_F=np.fft.fft((fermi_X*gsVal)[:])#[xPos,str,K]
                    integratedMat=np.tensordot((np.moveaxis([mixMat_F],0,1)*tCount_F),sigmaMat,axes=(2,0)) #[xpos,i,K]
                    tmpGs=bareCharge*pmAry-np.real(np.fft.ifft(integratedMat[:]))
                                    
                    diff=np.abs(tmpGs-gsVal)[:]
                    newVal=(tmpGs+gsVal)/np.float(2)
                    gsVal=newVal
                    stpNum+=1
                if np.amax(np.abs(diff))>np.amax(np.abs(diff2)):
                    stpNum+=20000
                           
                print(np.max(np.amax(np.abs(diff))),np.max(np.amax(np.abs(diff2))),stpNum)
                diff2=diff[:] 
            else:
                stpNum+=20000
        tempResult=gsVal*pmAry  
#        self.d_results['dressed_error']=diff
#        self.d_results['dressed']=tempResult

        return(np.asarray([tempResult,diff]))

    def dressChargeKer(self,bareKer,fermi_X,iterate=10,acc=10**-9):#same setup as above, could potentially be solved linearly, but works fine
        #used when fermi_X has an additional axis to the usual [str,rap] axes (for example an x-axis)
        #ie. fermi_X [xPos,string,rapidity]
        tCount_F= self.m_kernel #[str,str,K]
        stpNum=0; flipString=np.moveaxis(self.string,0,1)# 1 - string length 2 - parity 3 - auxiliary integers
        
        bShape=bareKer.shape;    dsc=bShape[-1];    #dK = 2*np.abs(mnX)/discretization; 
        sigmaMat=np.sign(flipString)[2]#auxiliary signs
        pmAry=np.array([np.ones(dsc//2),-np.ones(dsc//2)]).T.flatten(); #
    #    tCountShift_F=np.fft.ifftshift(tCount_F[:,:])#either shift or inverse shift can be used (identical for even discretizations)
        gsVal=bareKer*pmAry#[str,str,K]
        diff=np.abs(1); stpNum+=1; diff2=10**30
        while stpNum<10000: #ensuring convergence              
            if np.amax(np.abs(diff))>acc:
                for iterVal in np.arange(iterate):
                    mixMat_F=np.fft.fft((fermi_X*np.moveaxis(gsVal,0,1))[:])#[str,str,K]
#                    integratedMat=np.tensordot((np.moveaxis([mixMat_F],0,1)*tCount_F),sigmaMat,axes=(2,0)) #[i,j,K]
                    tCount_F_sh=np.reshape(mixMat_F,(1,bShape[0],bShape[0],bShape[-1]))
                    mixMat_sh=np.reshape(tCount_F,(bShape[0],bShape[0],1,bShape[-1]))
                    integratedMat=np.tensordot(mixMat_sh*tCount_F_sh,sigmaMat,axes=(1,0)) #[i,j,K]

                    tmpGs=bareKer*pmAry-np.real(np.fft.ifft(integratedMat[:]))
#                    diff=np.abs((tmpGs-gsVal)/gsVal)[:]
                                    
                    newVal=(tmpGs+gsVal)/np.float(2)
                    diff=np.abs((tmpGs-gsVal))[:]

                    gsVal=newVal
                    stpNum+=1
                if np.amax(np.abs(diff))>np.amax(np.abs(diff2)):
                    stpNum+=20000
                           
                print(np.max(np.amax(np.abs(diff))),np.max(np.amax(np.abs(diff2))),stpNum)
                diff2=diff[:] 
            else:
                stpNum+=20000
        tempResult=gsVal*pmAry  
#        self.d_results['dressed_error']=diff
#        self.d_results['dressed']=tempResult

        return(np.asarray([tempResult,diff]))

    def dressKer_X(self,bareKer,fermi_X):#same setup as above, could potentially be solved linearly, but works fine
        #used when fermi_X has an additional axis to the usual [str,rap] axes (for example an x-axis)
        #ie. fermi_X [xPos,string,rapidity]
        bShape=bareKer.shape;   # dsc=bShape[-1];    #dK = 2*np.abs(mnX)/discretization; 
#        tCount_X= self.m_kernel #[str,str,K]
#        summand=np.ones((bShape[0],bShape[-1]))
        stpNum=0; flipString=np.moveaxis(self.string,0,1)# 1 - string length 2 - parity 3 - auxiliary integers
        sigmaMat=np.reshape(np.sign(flipString)[2],(1,bShape[0],1,1))#auxiliary signs
        bShape=bareKer.shape;   # dsc=bShape[-1];    #dK = 2*np.abs(mnX)/discretization; 
#        pmAry=np.array([np.ones(dsc//2),-np.ones(dsc//2)]).T.flatten(); #
    #    tCountShift_F=np.fft.ifftshift(tCount_F[:,:])#either shift or inverse shift can be used (identical for even discretizations)
        driving=bareKer
        gsVal=bareKer#[str,str,1,K,K] ie. take the midpoint from the matrix vals
        fermi_use=np.reshape(fermi_X,(1,bShape[0],1,bShape[-1]))
        diff=np.abs(1); stpNum+=1; diff2=10**30
        while stpNum<10000: #ensuring convergence              
            if np.amax(np.abs(diff))>10**(-8):
                for iterVal in np.arange(10):
                    mixMat_X=fermi_use*bareKer#[str,str-Int,K,K-Int]
#                    integratedMat=np.tensordot((np.moveaxis([mixMat_F],0,1)*tCount_F),sigmaMat,axes=(2,0)) #[i,j,K]
                    integratedMat=self.m_disc*np.tensordot(mixMat_X*sigmaMat,gsVal,axes=((1,3),(0,3))) #[i,K,j,K]

                    tmpGs=driving-np.moveaxis(integratedMat,1,2)
#                    diff=np.abs((tmpGs-gsVal)/gsVal)[:]
                                    
                    newVal=(tmpGs+gsVal)/np.float(2)
                    diff=np.abs((tmpGs-gsVal))[:]

                    gsVal=newVal
                    stpNum+=1
                if np.amax(np.abs(diff))>np.amax(np.abs(diff2)):
                    stpNum+=20000
                           
                print(np.max(np.amax(np.abs(diff))),np.max(np.amax(np.abs(diff2))),stpNum)
                diff2=diff[:] 
            else:
                stpNum+=20000
        tempResult=gsVal 
#        self.d_results['dressed_error']=diff
#        self.d_results['dressed']=tempResult

        return(np.asarray([tempResult,diff]))




    def dressCharge(self,bareCharge,fermi_X,itrn=10):#same setup as above, could potentially be solved linearly, but works fine
        tCount_F=self.m_kernel
#        fermi_X = self.d_results['fermi_X_t0']
        stpNum=0; flipString=np.moveaxis(self.string,0,1)# 1 - string length 2 - parity 3 - auxiliary integers
        sigmaMat=np.sign(flipString)[2]#auxiliary signs
        bShape=bareCharge.shape;    dsc=bShape[-1];    #dK = 2*np.abs(mnX)/discretization; 
        pmAry=np.array([np.ones(dsc//2),-np.ones(dsc//2)]).T.flatten(); #tCount_F=np.fft.fft(pmAry*tCount_X[:,:])#[i,j,K+K0]   *np.reshape(sigmaMat,(1,stringNum,1))
    #    tCountShift_F=np.fft.ifftshift(tCount_F[:,:])#either shift or inverse shift can be used (identical for even discretizations)
        gsVal=bareCharge*pmAry
        diff=np.abs(1); stpNum+=1
        while stpNum<10000: #ensuring convergence        
            if np.amax(np.abs(diff))>5*10**(-12):
                for iterVal in range(1,itrn):
                    mixMat_F=np.fft.fft((fermi_X*gsVal)[:])
#                    integratedMat=np.tensordot(mixMat_F*tCount_F,sigmaMat,axes=(1,0)) #[i,K]
                    integratedMat=np.tensordot(mixMat_F*tCount_F,sigmaMat,axes=(1,0)) #[i,K]

                    tmpGs=bareCharge*pmAry-np.real(np.fft.ifft(integratedMat[:]))
                    diff=np.abs(tmpGs-gsVal)
                    newVal=(tmpGs+gsVal)/np.float(2)
                    gsVal=newVal
                    stpNum+=1
                print(np.max(np.amax(np.abs(diff))),stpNum)
            else:
                stpNum+=20000
        tempResult=gsVal*pmAry   
#        self.d_results['dressed_error']=diff
#        self.d_results['dressed']=tempResult

        return(np.asarray([tempResult,diff]))

#the dressing and initial energies should work, all that remains is to correct some smaller errors here
def fullEvalFourier(dsc,mnX,itrn,string,betaL,betaR,mu,top,btm,tCount_F):
    """Build the string"""
    dX=2*np.abs(np.float(mnX))/dsc;  stringNumber=len(string);   
    xList=np.linspace(mnX,np.abs(mnX),num=dsc,endpoint=False); alpha = top*np.pi/btm
    constants=np.asarray([dX,mnX,top*np.pi/btm,len(string),betaL,betaR,mu]); J=1;
    print("Finding the bare charges")
    stringVals=np.transpose(string)
    kPrime_X=2*np.pi*aCountCoord(alpha,xList,stringVals[0],stringVals[1])    
    e_X=-2*np.pi*J*(np.sin(alpha)/alpha)*aCountCoord(alpha,xList,stringVals[0],stringVals[1])
    ePrime_X=-2*np.pi*J*(np.sin(alpha)/alpha)*aCountPrimeCoord(alpha,xList,stringVals[0],stringVals[1])
    spin_X=np.tensordot(stringVals[0],np.ones(dsc),axes=0)
    print("Find Thermal Energy")
    thrmlE_X=yangYangE(constants,itrn,string,tCount_F,e_X) #compute via Yang-Yang eqn
    fermi_X=1/(1+np.exp(thrmlE_X[0]))
    print("Dressing Charges")    
    effePrime=dressCharge(constants,itrn,string,tCount_F,ePrime_X,fermi_X)
    effkPrime=dressCharge(constants,itrn,string,tCount_F,kPrime_X,fermi_X)
    effSpin=dressCharge(constants,itrn,string,tCount_F,spin_X,fermi_X)
    effectiveCharges=np.array([effePrime,effkPrime,effSpin])
    print("done dressing operation")
    print("primed dressed energy error: "+str(np.amax(np.abs(effectiveCharges[0][1]))))
    print("primed dressed momentum error: "+str(np.amax(np.abs(effectiveCharges[1][1]))))
    print("effective spin error: "+str(np.amax(np.abs(effectiveCharges[2][1]))))

    eDrPrime=effectiveCharges[0][0];    kDrPrime=effectiveCharges[1][0] #eff value of prime is prime of dressed value
    effDrSpin=effectiveCharges[2][0] # effective value of spin (string length)
    stringSign=np.sign(np.moveaxis(string,0,1))[2]
    density=np.reshape(stringSign,(stringNumber,1))*(kDrPrime*fermi_X)/(2.0*np.pi)
#    velocity=(eDrPrime/(kDrPrime+10**-20)+eDrPrime/(kDrPrime-10**-20))/2
    velocity=(eDrPrime/(kDrPrime))
    velocity.shape
    """Drude weight calculations"""
    fermiCorr=1-fermi_X; #gma=top*np.pi/btm
    spinSquared=(effDrSpin)*(effDrSpin)
    drudeNoSpin=fermiCorr*fermi_X*velocity*eDrPrime*np.sign(stringVals[2]).reshape(stringNumber,1)
    dW1=2*np.abs(mnX)/(dsc)*(drudeNoSpin.flatten()).dot(spinSquared.flatten())/(2*np.pi)
    
    velocitySquared=(velocity)*(velocity)*(density)*(fermiCorr); 
    velocitySquaredUsed=velocitySquared[:,np.int(dsc/4):np.int(3*dsc/4)+1]
    spinSquaredUsed=spinSquared[:,np.int(dsc/4):(np.int(3*dsc/4)+1)]
    dW2=2*np.abs(mnX)/(dsc)*(velocitySquared.flatten()).dot(spinSquared.flatten())
    dW3=2*np.abs(mnX)/(dsc)*(velocitySquaredUsed.flatten()).dot(spinSquaredUsed.flatten())
    listTemp=np.array([top,btm,dW1,dW2,dW3])
    setOfVals=np.array([eDrPrime,kDrPrime,effDrSpin,velocity,density,fermi_X])
    return(np.asarray([listTemp,setOfVals,string,constants,effectiveCharges[1]]))

#computes T function for a specific string and then compute the beta values
def fullEvalBeta(mnOBeta,mxOBeta,betaPts,top,btm,OmnX,Odsc):
    itrn=10
    string=xxzString.niceStringsFullQparity(top,btm)
    betaList=np.logspace(mnOBeta,mxOBeta,betaPts)#between 10^1stEntry - 10^2ndEntry take 12 evenly space points
    mnX=-2**OmnX; dsc=2**Odsc
    stringNumber=len(string); stringList=np.arange(stringNumber); #xList=np.linspace(mnX,np.abs(mnX),num=dsc,endpoint=False)
    mnK=np.pi*dsc/(2*mnX); kList=np.linspace(mnK,np.abs(mnK),num=dsc,endpoint=False)
    constants=np.asarray([dsc,mnX,top*np.pi/btm,len(string),0,0,0])
    tCount_F=np.zeros((stringNumber,stringNumber,dsc),np.float)
    for string1 in stringList:
        for string2 in stringList:
            tCount_F[string1,string2]=tKrnl_F(constants,string,string1,string2,kList)   
    for beta in betaList:
        tempHold=fullEvalFourier(dsc,mnX,itrn,string,beta,beta,10**(-6)/(beta),top,btm,tCount_F) 
        toSave=np.array([tempHold[0],tempHold[3]])
        np.save("anisotropy/drudeVals"+"/B"+ str(beta) + "top" + str(top)+"bottom"+str(btm)+"rap2raise"+str(OmnX)+"disc2raise"+str(Odsc)+"forBeta10raise"+str(mnOBeta) + "to10raise" + str(mxOBeta),toSave)

def drudeZeroT(top,bottom):
    gma=top *np.pi/float(bottom)
    return(np.pi *np.sin(gma)/(8*gma*(np.pi - gma)))
    
def drudeInftyT(top,bottom):
    tempDrude=(1/16.)*((np.sin(np.pi*top/np.float(bottom)))**2/(np.sin(np.pi/np.float(bottom)))**2)*(1 - (bottom/(2*np.pi))*np.sin(2*np.pi/np.float(bottom)))
    return(np.float(tempDrude))

def loadinDrudeValue(mnOBeta,mxOBeta,betaPts,top,btm,OmnX,Odsc):
#returns the value of beta, coefficient of Drude weight, full drude weight, difference wrt Prosen, difference wrt the zero
    betaList=np.logspace(mnOBeta,mxOBeta,betaPts);    
    drudeVals=np.array([np.load("anisotropy/"+str(top)+"over"+str(btm)+"/B"+ str(beta) + "top" + str(top)+"bottom"+str(btm)+"rap2raise"+str(OmnX)+"disc2raise"+str(Odsc)+"forBeta10raise"+str(mnOBeta) + "to10raise" + str(mxOBeta)+".npy")[0] for beta in betaList])
    drudeTemp=drudeVals[:,2];    diffProsen=(drudeInftyT(top,btm)-drudeTemp);    diffZero=(drudeZeroT(top,btm)-drudeTemp*betaList/(2))
    drudeSet=np.array([betaList,drudeTemp/2.,drudeTemp*betaList/2.,diffProsen,diffZero]) #all defined above
    return(drudeSet)

def loadinDrudeValueGen(mnOBeta,mxOBeta,betaPts,top,btm,OmnX,Odsc):
#returns the value of beta, coefficient of Drude weight, full drude weight, difference wrt Prosen, difference wrt the zero
    betaList=np.logspace(mnOBeta,mxOBeta,betaPts);    
    drudeVals=np.array([np.load("anisotropy/drudeVals"+"/B"+ str(beta) + "top" + str(top)+"bottom"+str(btm)+"rap2raise"+str(OmnX)+"disc2raise"+str(Odsc)+"forBeta10raise"+str(mnOBeta) + "to10raise" + str(mxOBeta)+".npy")[0] for beta in betaList])
    drudeTemp=drudeVals[:,2];    diffProsen=(drudeInftyT(top,btm)-drudeTemp);    diffZero=(drudeZeroT(top,btm)-drudeTemp*betaList/2)
    drudeSet=np.array([betaList,drudeTemp/2,drudeTemp*betaList/2,diffProsen,diffZero]) #all defined above
    return(drudeSet)


def singleEvalBeta(beta,top,btm,OmnX,Odsc):#useful to check that things will compute and making pictures
    itrn=10
    string=xxzString.niceStringsFullQparity(top,btm)
    mnX=-2**OmnX; dsc=2**Odsc
    stringNumber=len(string); stringList=np.arange(stringNumber); #xList=np.linspace(mnX,np.abs(mnX),num=dsc,endpoint=False)
    mnK=np.pi*dsc/(2*mnX); kList=np.linspace(mnK,np.abs(mnK),num=dsc,endpoint=False)
    constants=np.asarray([dsc,mnX,top*np.pi/btm,len(string),beta,0,0])
    tCount_F=np.zeros((stringNumber,stringNumber,dsc),np.float)
    for string1 in stringList:
        for string2 in stringList:
            tCount_F[string1,string2]=tKrnl_F(constants,string,string1,string2,kList)
    tempHold=fullEvalFourier(dsc,mnX,itrn,string,beta,beta,0,top,btm,tCount_F) 
    return(tempHold)

#def fermiWtEval(top,btm,tmx,tn,beta):
#    kmn=-np.pi*tn/(2*tmx); tmn = -tmx; J=1
#    tv=np.linspace(tmn,tmx,num=tn,endpoint=False); kv=np.linspace(kmn,np.abs(kmn),num=tn,endpoint=False)
#
#    stringVals=xxzString.niceStrings(top,btm).T #transpose so sVals = [np],[vp],[qp]
#    gma=top*np.pi/btm
#    constants=np.array([(tmx-tmn)/tn,tmn,gma,len(stringVals.T),beta,beta,0]); 
#    e_X=-2*np.pi*J*(np.sin(gma)/gma)*aCountCoord(gma,tv,stringVals[0],stringVals[1])
#
##    for i in np.arange(len(stringVals[0])):
##        plt.figure(11)
##        plt.plot(tv,e_X[i])
##        plt.show()
#    
#    stringNumber=len(stringVals.T); stringList=np.arange(stringNumber); #xList=np.linspace(mnX,np.abs(mnX),num=dsc,endpoint=False)
#    tCount_F=np.zeros((stringNumber,stringNumber,tn))
#    for string1 in stringList:
#        for string2 in stringList:
#            tCount_F[string1,string2]=tKrnl_F(constants,stringVals.T,string1,string2,kv)   
##    fermiWt=np.zeros((len(beta),stringNumber,tn))
##    for val in np.arange(len(beta)):
##        constants=np.asarray([np.abs(tmx-tmn)/tn,tmn,gma,len(stringVals.T),beta[val],beta[val],0]); 
##        thrmlE_X=xxz.yangYangE(constants,10,stringVals.T,tCount_F,e_X) #compute thermal energy via Yang-Yang entropy
##        fermiWt[val]=1/(1+np.exp(thrmlE_X[0]))
#    constants=np.array([np.abs(tmx-tmn)/tn,tmn,gma,len(stringVals.T),beta,beta,0]); 
#    thrmlE_X=yangYangE(constants,10,stringVals.T,tCount_F,e_X) #compute thermal energy via Yang-Yang entropy
#    fermiWt=1/(1+np.exp(thrmlE_X[0]))
#    btaFmi=np.array([beta,fermiWt.T[tn//2][-1]])
#    
#    for i in np.arange(len(fermiWt[0])):
#        plt.figure(10)
#        plt.plot(tv,fermiWt[i],'x')   
#        plt.show()  
#        
#    fitVel=np.polyfit(np.log(btaFmi[0]),np.log(-btaFmi[1]+fermiCtr(top,btm)),1)
##    fitVelVal=fitVel[1]+np.log(beta)*fitVel[0]
#    return(fitVel)
    
def onlyFmiWt(dsc,mnX,itrn,string,betaL,betaR,mu,top,btm,tCount_F):
    """Build the string"""
    dX=2*np.abs(np.float(mnX))/dsc;    
    xList=np.linspace(mnX,np.abs(mnX),num=dsc,endpoint=False); alpha = top*np.pi/btm
    constants=np.asarray([dX,mnX,top*np.pi/btm,len(string),betaL,betaR,mu]); J=1;
    print("Finding the bare charges")
    stringVals=np.transpose(string)
    e_X=-2*np.pi*J*(np.sin(alpha)/alpha)*aCountCoord(alpha,xList,stringVals[0],stringVals[1])
    print("Find Thermal Energy")
    thrmlE_X=yangYangE(constants,itrn,string,tCount_F,e_X) #compute via Yang-Yang eqn
    fermi_X=1/(1+np.exp(thrmlE_X[0]))
   
    return(fermi_X)
    
def evalFmiWt(beta,top,btm,OmnX,Odsc):#useful to check that things will compute and making pictures
    itrn=10
    string=xxzString.niceStringsFullQparity(top,btm)
    mnX=-2**OmnX; dsc=2**Odsc
    stringNumber=len(string); stringList=np.arange(stringNumber); #xList=np.linspace(mnX,np.abs(mnX),num=dsc,endpoint=False)
    mnK=np.pi*dsc/(2*mnX); kList=np.linspace(mnK,np.abs(mnK),num=dsc,endpoint=False)
    constants=np.asarray([dsc,mnX,top*np.pi/btm,len(string),beta,0,0])
    tCount_F=np.zeros((stringNumber,stringNumber,dsc),np.float)
    for string1 in stringList:
        for string2 in stringList:
            tCount_F[string1,string2]=tKrnl_F(constants,string,string1,string2,kList)
    tempHold=onlyFmiWt(dsc,mnX,itrn,string,beta,beta,0,top,btm,tCount_F) 
    return(tempHold)
