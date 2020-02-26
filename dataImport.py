"""
World Trade Model
data source: EXIOBASE ixi v.1.4 (2011)
@author: Matteo V. Rocco
"""

"""
URGENTI:
    - fare in modo che la matrice R e R_hh siano quelle originali di exiobase
    - rendere indipendente il codice dai file del database, che ognuno deve avere sul suo pc e che non saranno caricati su github perchè troppo pesanti


problemi/cose da fare:
    - non sono riuscito a definire il factor endowments includendo uso dei fattori esogeni fatto dagli households. 
    quest'ultimo è risutato a valle di matrice di allocazione e se lo includo il problema diventa non risolvibile.
    una strata può essere l'endogenizzazione dei consumi finali... però boh
    - da fare il problema duale
    - da fare il riconoscimento automatico se ci sono vincoli o meno. perchè ora se non ci sono vincoli e li lascio 
    non commentati, da errore
    - da fare tutta la parte di post-processamento
    - da fare la definizione degli endowments con criterio (per ora sono posti pari ai consumi all'anno base)
    - la soluzione (trade di prodotti tra settori/regioni) per ora è troppo scatterata e troppo concentrata sui primi
    paesi della lista. il che mi fa pensare a un possibile errore di definizione della matrice dei trade E (vedere excel
    di accompagnamento)

"""


import pandas as pd
import numpy as np
import cvxpy as cv


#%% DATA IMPORT
# importa tutti i dati dai .txt di exiobase
# definisce multi-index tramite indicazioni foglio excel indices.xlsx

"data indices"
dataFileName = "indices.xlsx"

sectorData = pd.read_excel(dataFileName,"sectorIndex", header=None, index_col=None, skiprows=1).values.T.tolist()
finalDemandData = pd.read_excel(dataFileName,"finalDemandIndex", header=None, index_col=None, skiprows=1).values.T.tolist()
exTransData = pd.read_excel(dataFileName,"exTransIndex", header=None, index_col=None, skiprows=1).values.T.tolist()

"main indices"
nC_dis,nC = len(list(set(sectorData[1]))),len(list(set(sectorData[7])))             # number of countries/aggregated countries
nI_dis,nI = len(list(set(sectorData[4]))),len(list(set(sectorData[10])))            # number of industries/aggregated industries per country
nY_dis,nY = len(list(set(finalDemandData[4]))),len(list(set(finalDemandData[10])))  # number of final demand/aggregated final demand categories
nR_dis,nR = len(list(set(exTransData[1]))),len(list(set(exTransData[5])))           # number of exogenous transactions categories


"multi-index definition"
sectorIndex = pd.MultiIndex.from_arrays(sectorData, names=["countryNumber", "countryCode", "countryName", 
                                                           "industryNumber", "industryCode", "industryName", 
                                                           "countryAggNumber", "countryAggCode", "countryAggName", 
                                                           "industryAggNumber", "industryAggCode", "industryAggName"])

finalDemandIndex = pd.MultiIndex.from_arrays(finalDemandData, names=["countryNumber","countryCode","countryName",
                                                                     "finalDemandNumber","finalDemandCode", "finalDemandName",
                                                                     "countryAggNumber", "countryAggCode", "countryAggName",
                                                                     "finalDemandAggNumber", "finalDemandAggCode", "finalDemandAggName"])

exTransIndex = pd.MultiIndex.from_arrays(exTransData, names=["exTransNumber","exTransCode","exTransName","exTransUnit",
                                                             "exTransAggNumber", "exTransAggCode", "exTransAggName",
                                                             "objFunction","fixedConst","sectorConst","regionConst","globalConst"])

"A - technical coefficients matrix"
A_values = pd.read_csv("A.txt", sep="\t", skiprows=2, usecols=[*range(2, nC_dis*nI_dis+2)]).values
A_raw = pd.DataFrame(A_values, index=sectorIndex, columns=sectorIndex)

"Y - final demand matrix (total values)"
Y_values = pd.read_csv("Y.txt", sep="\t", skiprows=2, usecols=[*range(2, nC_dis*nY_dis+2)]).values
Y_raw = pd.DataFrame(Y_values, index=sectorIndex, columns=finalDemandIndex)

"R, R_hh - exogenous transactions matrix (total values)"
R_values = pd.read_csv("R.txt", sep="\t", skiprows=1, usecols=[*range(1, nC_dis*nI_dis+1)]).values
R_values[15:25,:] = R_values[15:25,:] * 1e-6 # from kg to kton
R_values[29:33,:] = R_values[29:33,:] * 1e3/41868 # from TJ to ktoe
R_raw = pd.DataFrame(R_values, index=exTransIndex, columns=sectorIndex)

R_hh_values = pd.read_csv("R_hh.txt", sep="\t", skiprows=1, usecols=[*range(1, nC_dis*nY_dis+1)]).values
R_hh_values[15:25,:] = R_hh_values[15:25,:] * 1e-6 # from kg to kton
R_hh_values[29:33,:] = R_hh_values[29:33,:] * 1e3/41868 # from TJ to ktoe
R_hh_raw = pd.DataFrame(R_hh_values, index=exTransIndex, columns=finalDemandIndex)

#%% DATA PREPARATION AND AGGREGATION
# prepara i dati con multi-index e aggrega secondo indicazioni contenute nel foglio excel indices.xlsx
# controlla i bilanci economici output-outlays (da rivedere e fare meglio)

"derivation of total transactions matrix Z"
L_values = np.linalg.inv(np.eye(A_raw.shape[0]) - A_values)     # Leontief Inverse matrix
x = L_values @ np.sum(Y_values,1)                               # total production vector
xd = np.array([x]).T; xd[xd<=0]=1                               # doped total production vector
Z_values = A_values @ np.diagflat(xd)
Z_raw = pd.DataFrame(Z_values, index=sectorIndex, columns=sectorIndex)

"matrix aggregation"
Z = Z_raw.groupby(level=["countryAggNumber","countryAggCode","countryAggName","industryAggNumber","industryAggCode","industryAggName"],axis=0).sum()
Z = Z.groupby(level=["countryAggNumber","countryAggCode","countryAggName","industryAggNumber","industryAggCode","industryAggName"],axis=1).sum()
Y = Y_raw.groupby(level=["countryAggNumber","countryAggCode","countryAggName","industryAggNumber","industryAggCode","industryAggName"],axis=0).sum()
Y = Y.groupby(level=["countryAggNumber","countryAggCode","countryAggName","finalDemandAggNumber","finalDemandAggCode","finalDemandAggName"],axis=1).sum()
R = R_raw.groupby(level=["exTransAggNumber", "exTransAggCode", "exTransAggName","objFunction","fixedConst","sectorConst","regionConst","globalConst"],axis=0).sum()
R = R.groupby(level=["countryAggNumber","countryAggCode","countryAggName","industryAggNumber","industryAggCode","industryAggName"],axis=1).sum()
R_hh = R_hh_raw.groupby(level=["exTransAggNumber", "exTransAggCode", "exTransAggName","objFunction","fixedConst","sectorConst","regionConst","globalConst"],axis=0).sum()
R_hh = R_hh.groupby(level=["countryAggNumber","countryAggCode","countryAggName","finalDemandAggNumber","finalDemandAggCode","finalDemandAggName"],axis=1).sum()

"balance check DA SISTEMARE BENE"
output = np.sum(Z_values,1) + np.sum(Y_values,1)
outlays = np.sum(Z_raw.values,0) + np.sum(R_values[:9,:],0)

output_agg = np.sum(Z,1) + np.sum(Y,1)
outlays_agg = np.sum(Z,0) + np.sum(R.values[:9,:],0)
check_agg = (output_agg.sum() - outlays_agg.sum())/outlays_agg.sum()


#%% EXOGENOUS PARAMETERS
# prepara dati di base per WTM
# i fattori vanno commentati/uncommentati a seconda del fatto che li ho attivati o meno nel foglio excel indices
# se non ci sono factor endowments attivati, il codice da errore
# i factor endowments attivati sono fissati pari ai valori contenuti nella matrice R all'anno base
# definita mobilità fattori: 
#   - fixed: mobilità nulla. i fattori possono essere usati solo in un settore in una regione
#   - sector: mobilità intersettoriale. i fattori possono essere usati all'interno dello stesso settore in tutte le regioni
#   - region: mobilità interregionale. i fattori possono essere usati in tutti i settori ma senza uscire dalla regione
#   - global: mobilità globale. i fattori possono essere impiegati in tutti i settori e in tutte le regioni fino al limite globale

"basic IO data"
sectorIndexAgg = Z.index
finalDemandIndexAgg = Y.columns
exTransIndexAgg = R.index

x = Z.sum(1) + Y.sum(1)     # total production by sector, by region
xd = x.copy(); xd[xd<=0]=1  # doped total production vector by sector, by region

A = pd.DataFrame(Z.values @ np.linalg.inv(np.diagflat(xd.values)),index=sectorIndexAgg,columns=sectorIndexAgg)

"basi WTM data"
At = np.zeros([nI*nC,nI*nC])
Yt = np.zeros([nI*nC,nY*nC])
for c in range(nC):
    for r in range(nC):
        At[c*nI:c*nI+nI,c*nI:c*nI+nI] += A.values[r*nI:r*nI+nI,c*nI:c*nI+nI]
        Yt[c*nI:c*nI+nI,c*nY:c*nY+nY] += Y.values[r*nI:r*nI+nI,c*nY:c*nY+nY]
        
At = pd.DataFrame(At,index=sectorIndexAgg,columns=sectorIndexAgg)                                               # total technical coeff matrix
Yt = pd.DataFrame(Yt,index=sectorIndexAgg,columns=finalDemandIndexAgg)                                          # total final demand matrix
F = pd.DataFrame(R.values @ np.linalg.inv(np.diagflat(xd.values)),index=exTransIndexAgg,columns=sectorIndexAgg) # exogenous transactions coeff matrix

"factor endowments definition (factors mobility defined as fixed, by sector, by region, global)"
f_fixed = R.xs("x",level=4,drop_level=False).copy()
# f_sector = R.xs("x",level=5,drop_level=False).copy().groupby(level=["industryAggNumber","industryAggCode","industryAggName"],axis=1).sum()
f_region = R.xs("x",level=6,drop_level=False).copy().groupby(level=["countryAggNumber","countryAggCode","countryAggName"],axis=1).sum()
# f_global = R.xs("x",level=7,drop_level=False).copy().sum(1)


#%% PRIMAL PROBLEM
# definizione del problema primale. i vincoli sui constraint vanno attivati/disattivati a seconda che esistono (come sezione precedente)
# 

"variables"
x = cv.Variable((nC*nI,1),nonneg=True)      # total production (by sector, by country)

E = cv.Variable((nC*nI,nC),nonneg=True)     # trades of industry products (rows: products by sector, by country) among countries (columns)
E_ex = cv.sum(E,1,keepdims=True)            # exports (by sector, by country)
E_im = np.zeros([nI,nC])                    # imports (by sector, by country)
for i in range(nC):
    E_im += E[(i*nI):(i*nI+nI),:]
E_im = cv.reshape(E_im,(nC*nI,1))

"objective function"
objFunc = cv.matmul(np.sum(F.xs("x",level=3,drop_level=False).values,0,keepdims=True),x) # objective function: total exogenous economic factors
obj = cv.Minimize(objFunc)

"constraints definition"
const = [   cv.matmul(np.eye(nC*nI)-At.values,x) + E_im - E_ex == np.sum(Yt.values,1,keepdims=True),                                                                                        # final demand production balance (by sector, by country)
            *[cv.matmul(F.xs("x",level=4,drop_level=False).iloc[[r]],cv.diag(x)) <= f_fixed.iloc[[r]]],
            # *[cv.sum(cv.reshape(cv.matmul(F.xs("x",level=5,drop_level=False).iloc[[r]],cv.diag(x)),(nI,nC)),1,keepdims=True).T <= f_sector.iloc[[r]] for r in range(np.size(f_sector,0))],  # resources uses less than endowments per sector
            *[cv.sum(cv.reshape(cv.matmul(F.xs("x",level=6,drop_level=False).iloc[[r]],cv.diag(x)),(nI,nC)),0,keepdims=True) <= f_region.iloc[[r]] for r in range(np.size(f_region,0))],    # resources uses less than endowments per region
            # cv.sum(cv.matmul(F.xs("x",level=7,drop_level=False).values,cv.diag(x)),1) <= f_global,                                                                                          # resources uses less than endowments global
            
            *[E[r*nI:r*nI+nI,c] == 0 for r in range(nC) for c in range(nC) if r==c]                                                                                                         # diagonal matrix block in trade matrix E empty
        ]

"problem solution"
prob = cv.Problem(obj,const)
prob.solve(solver=cv.GUROBI,verbose=False)
print("status: ", prob.status)
print("optimal value: ", prob.value)


#%% DERIVATION OF TRANSACTION TABLES
# derivazione delle tabelle di transazione
# mi è capitato che in problema trovi una soluzione, ma che a questa non corrisponde alcuna matrice S (ovvero c'è un passaggio con matrice singolare)

S = np.zeros([nI*nC,nI*nC])

for c in range(nC):
    for r in range(nC):
        if r==c:
            S[nI*r:nI*r+nI,nI*c:nI*c+nI] = np.diagflat(np.linalg.inv(cv.diag(E_im[nI*r:nI*r+nI,:] + x[nI*r:nI*r+nI,:]).value) @ x[nI*c:nI*c+nI,:].value)
        else: 
            S[nI*r:nI*r+nI,nI*c:nI*c+nI] = np.diagflat(np.linalg.inv(cv.diag(E[nI*r:nI*r+nI,c:c+1] + x[nI*c:nI*c+nI,:]).value) @ E[nI*r:nI*r+nI,c:c+1].value)

x_res = pd.DataFrame(x.value,index=sectorIndexAgg)
A_res = pd.DataFrame(At.values @ S,index=sectorIndexAgg,columns=sectorIndexAgg)
Z_res = pd.DataFrame((At.values @ S) @ np.diagflat(x.value),index=sectorIndexAgg,columns=sectorIndexAgg)
Y_res = pd.DataFrame(S @ Yt.values,index=sectorIndexAgg,columns=finalDemandIndexAgg)
R_res = pd.DataFrame(F.values @ np.diagflat(x.value),index=exTransIndexAgg,columns=sectorIndexAgg)
R_hh_res = R_hh.copy()

E_res = pd.DataFrame(E.value,index=sectorIndexAgg)
E_im_res = pd.DataFrame(E_im.value,index=sectorIndexAgg)
E_ex_res = pd.DataFrame(E_ex.value,index=sectorIndexAgg)


#%% VARIABLES CLEANUP

del A_values,L_values,R_values,R_hh_values,Y_values,Z_values
del A_raw,R_raw,R_hh_raw,Y_raw,Z_raw
# del check,check_agg,output,outlays,output_agg,outlays_agg
del x,xd
del exTransData,exTransIndex,finalDemandData,finalDemandIndex,sectorData,sectorIndex
del nC_dis,nI_dis,nR_dis,nY_dis
del E,E_ex,E_im,S,c,r
del const,obj,dataFileName,i,prob


#%% RESULTS EXPORT AND ANALYSIS
# occorre fare molto lavoro di plot dei risultati

countryGDP = R_res.xs("Value Added",level=2,drop_level=False).groupby(level=["countryAggNumber","countryAggCode","countryAggName"],axis=1).sum()
test = R.xs("Value Added",level=2,drop_level=False).groupby(level=["countryAggNumber","countryAggCode","countryAggName"],axis=1).sum()






