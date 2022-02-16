#####################
#importing libraries#
#####################

from astropy.io import fits
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsp
import statistics
import copy
import scipy.stats as sstat


#figure settings
plt.rcParams["axes.titlesize"]="x-large"
plt.rcParams["axes.labelsize"]="medium"
plt.rcParams["legend.fontsize"]="small"
plt.rcParams["legend.loc"]="upper left"
plt.rcParams["legend.framealpha"]=1

######################
#functions definition#
######################

def checkexistence(maindir,subdir=""):
    home=os.getcwd()
    if not os.path.isdir(os.path.join(home,maindir)):
        os.mkdir(maindir)
        print("directory ",maindir," succesfully created")
    if subdir!="":
        if not os.path.isdir(os.path.join(home,maindir,subdir)):
            os.chdir(maindir)
            os.mkdir(subdir)
            os.chdir("..")
            print("directory ",subdir," succesfully created")
            
def clean(sample):
    #removing nan and inf values
    i=0
    while i<36:
        arr=sample[i]
        sample[i]=arr[np.isfinite(arr)]
        if i>6 and i<25:
            sample[i+1]=sample[i+1][np.isfinite(arr)]
            i+=2
        elif i>24 and i<31:
            sample[i+1]=sample[i+1][np.isfinite(arr)]
            sample[i+2]=sample[i+2][np.isfinite(arr)]
            i+=3
        else:
            i+=1

    #sigma-clipping
    N=4
    i=0
    while i<36:
        for j in range(7):
            arr=sample[i]
            if i>6 and i<25:
                sample[i+1]=sample[i+1][np.logical_and(arr>arr.mean()-N*arr.std(),arr<arr.mean()+N*arr.std())]
                sample[i]=arr[np.logical_and(arr>arr.mean()-N*arr.std(),arr<arr.mean()+N*arr.std())]
            elif i>24 and i<31:
                sample[i+1]=sample[i+1][np.logical_and(arr>arr.mean()-N*arr.std(),arr<arr.mean()+N*arr.std())]
                sample[i+2]=sample[i+2][np.logical_and(arr>arr.mean()-N*arr.std(),arr<arr.mean()+N*arr.std())]
                sample[i]=arr[np.logical_and(arr>arr.mean()-N*arr.std(),arr<arr.mean()+N*arr.std())]
            else:
                sample[i]=arr[np.logical_and(arr>arr.mean()-N*arr.std(),arr<arr.mean()+N*arr.std())]
        if i>6 and i<25:       
            i+=2
        elif i>24 and i<31:
            i+=3
        else:
            i+=1
    return sample

def adjust_on_y(x,y):
    #removing nan and inf values
    x=x[np.isfinite(y)]
    y=y[np.isfinite(y)]
    
    #sigma-clipping
    N=4
    for j in range(7):
        x=x[np.logical_and(y>y.mean()-N*y.std(),y<y.mean()+N*y.std())]
        y=y[np.logical_and(y>y.mean()-N*y.std(),y<y.mean()+N*y.std())]
    
    return x,y

def adjust_on_x(x,y):
    #removing nan and inf values
    y=y[np.isfinite(x)]
    x=x[np.isfinite(x)]
    
    #sigma-clipping
    N=4
    for j in range(7):
        y=y[np.logical_and(x>x.mean()-N*x.std(),x<x.mean()+N*x.std())]
        x=x[np.logical_and(x>x.mean()-N*x.std(),x<x.mean()+N*x.std())]
    
    return x,y

def gauss(bins,media,sigma):
    x=np.zeros(len(bins)-1)
    for i in range(len(x)):
        x[i]=(bins[i]+bins[i+1])/2
    return x, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-media)**2/(2*sigma**2))

#1d plots function 
def genhist(arr,name,sample,nbins=50):
    if name=="z":       #estimation of mean and standard deviation without python standard modules
        mean=0
        for i in arr:
            mean+=i
        mean=mean/len(arr)
        std=0
        for i in arr:
            std+=(i-mean)**2
        std=np.sqrt(std/(len(arr)-1))
    else:               #assignment of mean and standard deviation with python standard modules
        mean=arr.mean()
        std=arr.std()
    median=statistics.median(arr)
    meanerr=std
    medianerr=1.4826*statistics.median(np.absolute(arr-median))
    
    #creating subplots
    plt.figure(1,figsize=(10,5))
    fig=gsp.GridSpec(1,3,wspace=0.5,hspace=0.25)
    #mean-histogram generation
    axhist1=plt.subplot(fig[0])
    counts,bins,ign=axhist1.hist(arr,nbins,density=True)
    p1=plt.axvline(mean,c="k",ls="--")
    p2=plt.axvspan(mean-meanerr,mean+meanerr,color="g",alpha=0.25)
    plt.xlabel(name)
    axhist1.set_title("mean_plot")
    #plotting gaussian
    x,mod=gauss(bins,mean,std)
    p3,=axhist1.plot(x,mod,c="g",ls="--",lw=2)
    axhist1.legend([p1,p2,p3],["mean","error","gaussian"])
    #median-histogram generation
    axhist2=plt.subplot(fig[2])
    axhist2.hist(arr,nbins,density=True)
    p1=plt.axvline(median,c="k",ls="--")
    p2=plt.axvspan(median-medianerr,median+medianerr,color="r",alpha=0.25)
    plt.xlabel(name)
    axhist2.set_title("median_plot")
    axhist2.legend([p1,p2],["median","error"])

    #file output
    res = np.zeros(1, dtype=[('var1', 'U29'), ('var2', float), ('var3', float), ('var4', float), ('var5', float)])
    res["var1"]=name
    res["var2"]=mean
    res["var3"]=std
    res["var4"]=median
    res["var5"]=medianerr
    filename="output/"+sample+".dat"
    f=open(filename,"a")
    f.write(os.linesep)
    np.savetxt(f,res,delimiter=" ",fmt="%s   %f   %f   %f   %f",newline=os.linesep)
    f.close()
    return x,counts,mod,fig

#residuals function
def genresid(axid,arr,counts,mod,name,sample):
    ax=plt.subplot(axid)
    resid=(counts-mod)
    ax.scatter(arr,resid)
    ax.axhline(resid.mean(),c="0.3",ls="--")
    ax.axhspan(resid.mean()-resid.std(),resid.mean()+resid.std(),color="0.3",alpha=0.25)
    ax.set_title("residuals_plot")
    title="plot/"+sample+"/"+name+".png"
    plt.savefig(title,dpi=150)
    plt.clf()

#2d plots function
def gencorr(arr1,arr2,sample,name):
    arr1,arr2=adjust_on_y(arr1,arr2)
    arr1,arr2=adjust_on_x(arr1,arr2)
    plt.figure(1,figsize=(10,5))
    fig=gsp.GridSpec(1,2,wspace=0.25,hspace=1)
    deg=[0,1]
    fits=["constant","linear"]
    for c in range(len(deg)):
        #fit
        res=np.polyfit(arr1,arr2,deg[c])
        y_bf=np.polyval(res,arr1)
        #creating subplots
        ax=plt.subplot(fig[c])
        p1=ax.scatter(arr1,arr2,edgecolor="k")
        p2,=ax.plot(arr1,y_bf,c="orange",ls="--",linewidth=2.5)
        plt.xlabel("redshift")
        plt.ylabel(name)
        ax.set_title(fits[c]+" correlation")
        ax.legend([p1,p2],["data",fits[c]+" fit"])
    title="plot/"+sample+"/z_"+name+".png"
    #searching for correlations
    r=sstat.pearsonr(arr1,arr2)
    if np.absolute(r[0])>=0.5:
        #creation of directory for plots from redshift bins
        s="z_correlation_w_"
        os.chdir("plot")
        checkexistence(sample,s)
        os.chdir("..")
        #appending quantity name on file
        filename="output/"+sample+".dat"
        f=open(filename,"a")
        f.write(os.linesep+"#"+s+name)
        f.close()
        #drawing bins on figure1
        for k in range(5):
            n="bin_"+str(k+1)
            if k!=0:
                ax.axvline(0.02*k,c="k",ls="--")
            plt.text(0.1+0.2*k,0.01,n,ha="center",transform=ax.transAxes)
        plt.savefig(title,dpi=150)
        plt.clf()
        #plotting bins on figure2
        for k in range(5):
            n="bin_"+str(k+1)
            ybins=arr2[np.logical_and(arr1<0.02*(k+1),arr1>0.02*k)]
            a,counts,mod,fig=genhist(ybins,n,sample,50)
            genresid(fig[1],a,counts,mod,s+"/"+name+"-"+n,sample)
    else:
        plt.savefig(title,dpi=150)
        plt.clf()
        
#color plots function
def gencol(arr1,arr2,carr,n1,n2,name):
    os.chdir("plot")
    checkexistence("diagrams",name)
    os.chdir("..")
    #cleaning data
    savearr2=copy.deepcopy(arr2)
    arr1,arr2=adjust_on_y(arr1,arr2)
    
    savearr1=copy.deepcopy(arr1)
    carr,savearr2=adjust_on_y(carr,savearr2)
    
    arr1,arr2=adjust_on_x(arr1,arr2)
    savearr1,carr=adjust_on_x(savearr1,carr)
    #producing theorical relation
    sortedindexes=arr1.argsort()
    carr=carr[sortedindexes]
    arr2=arr2[sortedindexes]
    arr1=arr1[sortedindexes]
    y_rel=[]
    rel=""
    if name=="BPT":
        #mask on x
        carr=carr[arr1<0.038]
        arr2=arr2[arr1<0.038]
        arr1=arr1[arr1<0.038]
        y_rel=0.61/(arr1-0.05)+1.3
        #mask on y
        carr=carr[y_rel>-1.5]
        arr2=arr2[y_rel>-1.5]
        arr1=arr1[y_rel>-1.5]
        y_rel=y_rel[y_rel>-1.5]

        a=""
        rel=r'log([OIII]/Hβ) = 0.61/(log([NII]/Hα) - 0.05) + 1.3'
        xlab=r'log([NII]/Hα)'
        ylab=r'log([OIII]/Hβ)'
    elif name=="color-mass":
        y_rel=-0.495+0.25*arr1
        rel=r'(u − r) = −0.495 + 0.25 ∗ log10(M/M⊙)'
        xlab=r'log10(M/M⊙)'
        ylab=r'u − r'
    elif name=="SFR-mass":
        y_rel=-8.64+0.76*arr1
        rel=r'SFR = −8.64 + 0.76 ∗ log10(M/M⊙)'
        xlab=r'log10(M/M⊙)'
        ylab=r'SFR'
        
    #generating plot
    plt.figure(1,figsize=(10,5))
    fig=gsp.GridSpec(1,1,wspace=0.25,hspace=0.25)
    ax=plt.subplot(fig[0])
    scatter=ax.scatter(arr1,arr2,c=carr,cmap="copper")
    cb=plt.colorbar(scatter)
    cb.set_label("redshift")
    p2,=ax.plot(arr1,y_rel,c="k",ls="--",linewidth=2.5)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax.set_title(name+"_diagram")
    ax.legend([scatter,p2],["data",rel])
    title="plot/diagrams/"+name+"/diagram.png"
    plt.savefig(title)
    plt.clf()
    #generating histograms
    xinf=arr1[arr2<y_rel]
    xsup=arr1[arr2>y_rel]
    yinf=arr2[arr2<y_rel]
    ysup=arr2[arr2>y_rel]
    l=[xinf,xsup,yinf,ysup]
    n=[n1+"_blw_relation",n1+"_abv_relation",n2+"_blw_relation",n2+"_abv_relation"]
    filename="output/diagrams.dat"
    f=open(filename,"a")
    f.write(os.linesep+"#"+name)
    f.close()
    for i in range(4):
        a,counts,mod,fig=genhist(l[i],n[i],"diagrams",50)
        genresid(fig[1],a,counts,mod,name+"/"+n[i],"diagrams")
        
    
#main function for figures
def genplot(x,headers,j,tplot,arr1index=6):
    filename="output/"+j+".dat"
    f=open(filename,"w")
    f.write("#quantity  mean  std  median  err_mediana")
    f.close()
    i=6
    while i<36:
        if tplot=="hist":
            a,counts,mod,fig=genhist(x[i],headers[i],j)
            genresid(fig[1],a,counts,mod,headers[i],j)
        elif tplot=="fit" and i!=arr1index and ((i>14 and i<18)or (i>24 and i<32)):
            gencorr(x[arr1index],x[i],j,headers[i])
        if i>6 and i<25:
            i+=2
        elif i>24 and i<30:
            i+=3
        else:
            i+=1
            
###########
#main code#
###########


print("WARNING: the following warning messages are caused by the use of deprecated modules; the code runs properly despite them")

#organizing folders

dirname="project_SimonePucci"
checkexistence(dirname)
os.chdir(dirname)
checkexistence("data")
checkexistence("output")

home=os.getcwd()
ls=["parentsample","subsample","subsample_z","diagrams"]
for i in ls:
    checkexistence("plot",i)

if not os.path.isfile(os.path.join(home,"data","data_SDSS_Info.fit")):
    os.chdir("..")
    shutil.move("data_SDSS_Info.fit",os.path.join(home,"data"))
#uncomment the following line if using Linux and use that rather than shutil
    #os.system("mv data_SDSS_Info.fit project_SimonePucci/data")
    print("file 'data_SDSS_Info.fit' succesfully moved to folder 'data'")
    os.chdir(dirname)

#opening file
hdul=fits.open("data/data_SDSS_Info.fit")
data=hdul[1]
cols=data.columns
headers=cols.names

print("column_name\tindex")
c=0
for i in headers:
    print(i," ",c)
    c+=1

values=data.data
hdul.close()
    
#reading sample
parentdata=[]
for i in headers:
    parentdata.append(values[i])
    
#reading subsample
myid=16
mask=[values["ID"]==myid]
subdata=[]
for i in headers:
    subdata.append(values[i][mask])
    
subdataz=copy.deepcopy(subdata)

#cleaning samples
parentdata=clean(parentdata)
subdata=clean(subdata)
#array conversion
parentdata=np.array(parentdata)
subdata=np.array(subdata)
subdataz=np.array(subdataz)

#generating histograms
genplot(parentdata,headers,"parentsample","hist")
genplot(subdata,headers,"subsample","hist")
genplot(subdataz,headers,"subsample_z","fit",6)

#step IV
#getting data
nii=values["nii_6584_flux"][mask]
oii=values["oiii_5007_flux"][mask]
bptx=np.log10(nii/(values["h_alpha_flux"][mask]))
bpty=np.log10(oii/(values["h_beta_flux"][mask]))
mass=values["lgm_tot_p50"][mask]
ur=values["petroMag_u"][mask]-values["petroMag_r"][mask]
sfr=values["sfr_tot_p50"][mask]
#creating output file
filename="output/diagrams.dat"
f=open(filename,"w")
f.write("#quantity  mean  std  median  err_mediana")
f.close()
#generating plots
names=["BPT","color-mass","SFR-mass"]
ns=[["n_alpha_flux","o_beta_flux"],["mass","u_r_Mag"],["mass","sfr"]]
variables=[[bptx,bpty],[mass,ur],[mass,sfr]]
for i in range(3):
    gencol(variables[i][0],variables[i][1],values["z"][mask],ns[i][0],ns[i][1],names[i])

