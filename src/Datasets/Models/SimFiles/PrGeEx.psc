# Prokarotic Gene Expression
# Francesca Cairoli

Modelname: PrGeEx

# Set model to run with numbers of individuals
Species_In_Conc: False
Output_In_Conc: False

# Differential Equations as Reactions
R1:
	PLac+RNAP > PLacRNAP
	k1*PLac*RNAP
	
R2:
	PLacRNAP > PLac+RNAP
	k2*PLacRNAP

R3:
	PLacRNAP > TrLacZ1
	k3*PLacRNAP
	
R4:
	TrLacZ1 > RbsLacZ+PLac+TrLacZ2
	k4*TrLacZ1
	
R5:
	TrLacZ2 > RNAP
	k5*TrLacZ2
	
R6:
	Ribosome + RbsLacZ > RbsRibosome
	k6*Ribosome*RbsLacZ
	
R7:
	RbsRibosome > Ribosome + RbsLacZ
	k7*RbsRibosome
	
R8:
	RbsRibosome > TrRbsLacZ + RbsLacZ
	k8*RbsRibosome
	

R9:
	TrRbsLacZ > LacZ
	k9*TrRbsLacZ
	
R10:
	LacZ > dgrLacZ
	k10*LacZ
	
R11:
	RbsLacZ > dgrRbsLacZ
	k11*RbsLacZ
	
# Parameter Values

k1 = 0.17
k2 = 10
k3 = 1
k4 = 1
k5 = 0.015
k6 = 0.17
k7 = 0.45
k8 = 0.4
k9 = 0.015
k10 = 0.0000642
k11 = 0.3

PLac = 1
RNAP = 35
PLacRNAP = 0
TrLacZ1 = 0
RbsLacZ = 0
TrLacZ2 = 0
Ribosome = 350
RbsRibosome = 0
TrRbsLacZ = 0
LacZ = 0
dgrLacZ = 0
dgrRbsLacZ = 0


