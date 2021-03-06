# Susceptible-Infected-Recovered dynamic disease transmission model
# PyCSeS Implementation
# Author: Francesca Cairoli

Modelname: SIR
Description: PySCes Model Description Language Implementation of SIR model

# Set model to run with numbers of individuals
Species_In_Conc: False
Output_In_Conc: False

# Differential Equations as Reactions
R1:
	S+I > I+I
	beta*S*I/(S+I+R)
	
R2:
	I > R
	gamma*I

# Parameter Values
S = 95
I = 5
R = 0
beta = 0.12
gamma = 0.05

# Total population size, N
#!F N = S+I+R 
