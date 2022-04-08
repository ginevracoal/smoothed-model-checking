Modelname: PRDeg
Description: Phosphorelay network with B degradation

# Set model to run with numbers of individuals
Species_In_Conc: False
Output_In_Conc: False


# Differential Equations as Reactions
R1:
	$pool > B
	kprod
	
R2:
	L1+B > L1P+B
	k1*L1*B/N

R3:
	L1P+L2 > L1+L2P
	k2*L1P*L2/N

R4:
	L2P+L3 > L2+L3P
	k3*L2P*L3/N

R5:
	L3P > L3
	k4*L3P/N

R6:
	B > $pool
	kdeg*B
	
	
# Parameter values
L1P = 0
L2P = 0
L3P = 0
L1 = 32
L2 = 32
L3 = 32
B = 0
N = 5000

kprod = 0.1
kdeg = 0.05
k1 = 1
k2 = 1
k3 = 1
k4 = 2
