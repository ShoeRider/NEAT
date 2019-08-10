
#NEAT (Neruological Evolution through Augmenting Topologies)
import random

class NEAT_Node():
    def __init__(Innovation):
        self.InnovationNumber = Innovation
        self.edgeNumber = 0
        
class NEAT_Edge():
    def __init__(Innovation):
        self.InnovationNumber = Innovation
        self.weight = random.randint(-10,10)
        self.From = 0
        self.To = 0
        
        

    
#Species
class Species()
    def __init__(self,MutationChance=0.8,AddNewNode=0.2,ChangeWeight=0.4):
        self.MutationChance = MutationChance
        self.AddNewNode     = AddNewNode
        self.ChangeWeight   = ChangeWeight
        
        # Mutation Chances:
        self.MutateEdge     = 0
        self.MutateNode     = 0
        
        self.AddEdge        = 0
        self.MutateEdge     = 0
        self.AddNode        = 0
        self.MutateNode     = 0
        
        self.Nodes = []
        self.Edges = []

        self.InputSize  = 0
        self.Input      = []
        self.OutputSize = 0
        self.Output     = []

        self.NonModifiableNodes = self.OutputSize + self.InputSize
        
    def Mutation(self):
        #Create New Edge
        if random.random() > self.MutateEdge:
            break
        
        #Create New Node 
        if random.random() > self.MutateNode:
            
            break
        return 0

    def Simulate(self,Input):
        return 0
    def Copy(self):
        return copy.deepcopy(self)
    
    # Determine compatibility:
        #    - Excess/Disjoint genes, gene wieghts
        #D~ = ((C1)E / N) + ((C2)D/N) + C3 W^(Hat)
    def FindDistance(GenomeGenes,Species1,C1=0,C2=0,C3=0):
        #for  in Sp
        #Find Disjoint Genes
        # find the average weight of matching genes
        MatchingGenes = 0
        DisjointGenes = 0
        MatchingWeightSum = 0
        
        for Genes:
            if matching:
                MatchingGenes+=1
                MatchingWeightSum += abs(Edge.Weight)
            else:
                DisjointGenes+=1
                
        AverageWeightDifference = MatchingWeightSum/MatchingGenes
        #()/GenomeGenes + /GenomeGenes + C3 * AverageWeightDifference
        return 0
    
    def Reproduce(self,Species1):
        
        return 0

        
class NEAT():
    def __init__(self, **kwargs):
        
        self.PreGame         = PreGame
        self.GameDecorator   = GameDecorator
        self.PostGame        = PostGame
        self.InnovationCount = 0

        self.C1 = kwargs["C1"]
        self.C2 = C2
        self.C3 = C3

        self.DeathRate      = 0.5
        


        
        # Note Max Species = MaxGenomeSize * MaxGenomes
        self.MaxGenomeSize = 0
        self.MaxGenomes    = 0
        
        # Population control: distribute genome population across species.
        # Overall Population cap
        # Species Population Cap

        # Fitness function consider:
        #  - Final game Score
        #  - Gene Penalty (eventually with shifting value)

        # Stagnation:
        #   - Stagnation: remove because of lack of innovation
        #   - Extinction: remove because population is too low to servive

        


        #Challenges:
        # inputs
        # fitness function
        # Encoding
        # Parameter tuning
        self.MutationChance = 0.8
        self.DistanceMatrix = []
        self.EntireGenome   = []

    def Simulate(self):

        for Species in self.EntireGenome:
            if self.SimulatePreGame == NULL:
                break
            if self.SimulateGame() == NULL:# FunctionList
                break
            if self.NaturalSelection() == NULL:
                break
        self.NaturalSelection()
        self.Mutation()

        

    
    def NaturalSelection(self):
        #1. Separate species into different genomes
        Genomes = [] #Where [genome][Species]
        
        #Take all species, and find the distance between every single genome
        if Distance>Threashold:
        
            #species belongs in different genome

        #2. Natural Selections
        NewEntireGenome = []
        for Genome in Gemones:
            self.GenomeNaturalSelection(Genome)

            #take the remaining Species and add them to the real remaining list
            for Species in Genome:
                NewEntireGenome.append(Species)
        self.EntireGenome = NewEntireGenome
                
    #Where Genome is a list of the different species closeist to eachother.
    def GenomeNaturalSelection(self,Genome):
        if len(Genome) == 1:
            Genome.remove(Genome[0])
        KillOff = int(len(Genome)/2)
        
        SortBasedOnFitness(Genome)
        
        #Remove Lower Preforming Species
        Genome = Genome[:KillOff]
        
        
    def SortBasedOnFitness(self,Genome):
        for Index in range(len(Genome)):
            minimumIndex = Index
            for InnerIndex in range(Index):
                if Genome[InnerIndex].Fitness > Genome[minimumIndex].Fitness:
                    minimumIndex = InnerIndex
            Temp                 = Genome[Index]
            Genome[Index]        = Genome[minimumIndex] 
            Genome[minimumIndex] = Temp



    
    def Mutation(self,Species):
        for Species in self.EntireGenome:
            if random.random() > self.MutationChance:
                Species.Mutation(self.InnovationCount)
                self.InnovationCount += 1
                
                
            




#GA - Tests:
#   Fixed Topology
#   Random
