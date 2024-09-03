
import csv
import matplotlib.pyplot as plt
import math
import numpy as np

EARTH_RADIUS = 6371e3

class Tower:
    def __init__(self, id, easting, northing, long, lat):
        self.id = id
        self.easting = easting
        self.northing = northing
        self.long = long
        self.lat = lat
        self.frequency = None
        self.edges = []
        self.outer = None
        
def importData(file):
    towers = []
    
    with open(file, 'r') as csvFile:
        reader = csv.DictReader(csvFile)
        
        for row in reader:
            tower = Tower(
                id=row['ID'],
                easting=float(row['Easting']),
                northing=float(row['Northing']),
                long=float(row['Long']),
                lat=float(row['Lat'])
            )
            towers.append(tower)
    
    return towers

def calculateDistance(tower1, tower2):
    phi1 = math.radians(tower1.lat)
    phi2 = math.radians(tower2.lat)
    deltaPhi = math.radians(tower2.lat - tower1.lat)
    deltaLambda = math.radians(tower2.long - tower1.long)
    
    a = ( math.sin(deltaPhi/2) * math.sin(deltaPhi/2) 
        + math.cos(phi1) * math.cos(phi2) 
        * math.sin(deltaLambda/2) * math.sin(deltaLambda/2) )
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    d = EARTH_RADIUS * c
    
    return d

def calculateDistancePlanar(tower1, tower2):
    x = tower2.easting - tower1.easting
    y = tower2.northing - tower1.northing
    
    distance = math.sqrt(x**2 + y**2)
    
    return distance
         

def kNN(towerList, n):
    length = len(towerList)
    for i in range(length):
        distances = dict()
        sorted_distances =[]
        for j in range(length):
            if not (j == i):
                distance = calculateDistance(towerList[i], towerList[j])
                distances[towerList[j]] = distance
        
        sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        towerList[i].edges = list(sorted_distances.keys())[:n]
        
        nearDistances = list(sorted_distances.values())[n:]
        
        
        count = 0
        for k in range(len(nearDistances)):
            if (nearDistances[k] < 600):
                towerList[i].edges.append(list(sorted_distances.keys())[k+n])
                #print(list(sorted_distances.keys())[i+n].id)
                count = count +1
        
        towerList[i].outer = list(sorted_distances.keys())[n+count]    
        
        
def assignFrequencies(towerList):
    frequencies = list(range(110, 116)) #110 to 115
    
    for tower in towerList:
        usedFrequencies = set()
        for edge in tower.edges:
            if edge.frequency != None:
                usedFrequencies.add(edge.frequency)
        
        print()
        #print(tower.outer.id)
        #if len(usedFrequencies) < 2:
        #    print("beep")
        #    usedFrequencies.add(tower.outer.frequency)    
            
        availableFrequencies = []
        
        for f in frequencies:
            if f not in usedFrequencies:
                availableFrequencies.append(f)
                
        #available_frequencies = [f for f in frequencies if f not in usedFrequencies]
        print() 
        print(tower.id, availableFrequencies)
        
        
        for t in tower.edges:
            print(t.id, t.frequency)
        if availableFrequencies:
            tower.frequency = availableFrequencies[0]
        else:
            tower.frequency = frequencies[0]  # Assign first frequency if no other option

def calculateBasePerformance(towerList):
    sum = 0
    for tower in towerList:
        sum = sum + calculateDistance(tower, tower.edges[0])
        
    return sum / len(towerList)   

def calculateAverageAllocationDistance(towerList):
    sum = 0
    smallestFreq = 5000
    for freq in range(110, 116):
        sumFreq =0
        towers_with_freq = [t for t in towerList if t.frequency == freq]
        
        if (len(towers_with_freq)==0):
            break
        
        print(freq)
        length = len(towers_with_freq)
        for i in range(length):
            distances = dict()
            sorted_distances =[]
            #print(towers_with_freq[i].id)
            if (len(towers_with_freq) == 1):
                break
            for j in range(length):
                if not (j == i):
                    distance = calculateDistance(towers_with_freq[i], towers_with_freq[j])
                    distances[towerList[j]] = distance
                    
                    if (distance < smallestFreq):
                        smallestFreq = distance

            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            #print(list(sorted_distances.values())[0])
            sum = sum + list(sorted_distances.values())[0]
        
        #avgFreq = sumFreq/length
        #sum = sum + sumFr
    
    print('smallest distance= ', smallestFreq)
    
    return sum/len(towerList)
            
def calculateAverageAllocationDistance2(towerList):
    sum = 0
    smallestFreq = 5000 #To determine the smallest pair distance   
    
    for freq in range(110, 116):
        sumFreq =0
        singleTowerFrequencies =0
        towersAtFreq = []
        for t in towerList:
            if (t.frequency == freq):
                towersAtFreq.append(t)
        
        if (len(towersAtFreq)==0): #Break if no towers in frequency band
            break
        
        if (len(towersAtFreq)==1): #Break if only one tower in frequency band
            singleTowerFrequencies = singleTowerFrequencies +1
            break
        
        length = len(towersAtFreq)
        for i in range(length):
            distances = dict()
            sorted_distances =[]

            if (len(towersAtFreq) == 1):
                break
            for j in range(length):
                if not (j == i):
                    distance = calculateDistance(towersAtFreq[i], towersAtFreq[j])
                    distances[towerList[j]] = distance
                    
                    if (distance < smallestFreq):
                        smallestFreq = distance    

            sorted_distances = dict(sorted(distances.items(), key=lambda item: item[1]))
            sum = sum + list(sorted_distances.values())[0]
        
    
    print('smallest distance= ', smallestFreq)
    
    return sum/(len(towerList)-singleTowerFrequencies) #Only divide by the number towers with frequency pair  
            


def plot_towers(towers):
  
    longitudes = [tower.long for tower in towers]
    latitudes = [tower.lat for tower in towers]
    

    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes, latitudes, alpha=0.5)
    plt.title('Tower Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    

    for tower in towers:
        plt.annotate(tower.id, (tower.long, tower.lat), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8)
    

    plt.grid(True)
    plt.show()


def plot_towers_with_frequencies(towerList):
    #plt.figure(figsize=(12, 8))
    for frequency in range(110, 116):
        towers_with_freq = [t for t in towerList if t.frequency == frequency]
        longitudes = [t.long for t in towers_with_freq]
        latitudes = [t.lat for t in towers_with_freq]
        plt.scatter(longitudes, latitudes, label=f'Frequency {frequency}')

    plt.title('Tower Locations with Assigned Frequencies')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    
    for tower in towerList:
        plt.annotate(tower.id, (tower.long, tower.lat), xytext=(5, 5), 
                     textcoords='offset points', fontsize=8)
    
    plt.grid(True)
    plt.show()
       
     
                        
            
            
#########################################################################################################################################


file = 'cellTowerData.csv'
towerList = importData(file)

    
for tower in towerList[:5]:
    print(f"Tower ID: {tower.id}, Easting: {tower.easting}, Northing: {tower.northing}, Longitude: {tower.long}, Latitude: {tower.lat}")




kNN(towerList, 5)

assignFrequencies(towerList)

plot_towers_with_frequencies(towerList)

###############################################################################################################################################


print(calculateBasePerformance(towerList), calculateAverageAllocationDistance2(towerList))

criticalD = np.array([4.9, 140.6, 179.6, 179.6, 270.8, 238.2, 320.9, 270.9, 270.9, 108.8, 
                      108.8, 108.8, 108.8, 138.8, 138.8, 138.8, 108.8, 108.8, 108.8, 108.8, 108.8])

distance = np.array([161.5, 288.1, 389.8, 492.2, 542.9, 565.4, 611, 657.2, 
                     657.2, 605.9, 615.4, 513.7, 569.2, 495.4, 523.2, 542.7, 392.2, 173.2, 173.2, 173.2, 173.2])


distanceO = np.array([161.5, 455.9, 459.6, 595, 691.7, 675.5, 651.5, 657.1, 657.1, 605.9, 615.4, 5422.2, 600.8, 385, 379 ])
criticalO = np.array([4.9, 353.8, 353.8, 353.8, 353.8, 353.8, 270.8, 270.8, 270.8, 108.8, 108.8, 108.8, 108.8, 108.8, 108.8 ])

radius = np.array([250, 300, 350, 400, 450, 500, 550, 600 ])
radiusDistance = np.array([565.2, 586, 675, 665, 665, 665, 615, 697])
minDistance = np.array([238, 321, 353, 270, 270, 270, 270, 244 ])


n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

##########################################################################################################
#plotting metrics
##########################################################################################################

plt.plot(n, distance, 'b-o')

# Find the highest point
max_index = np.argmax(distance)
max_n = n[max_index]
max_distance = distance[max_index]

# Label the highest point
plt.annotate(f'({max_n}, {max_distance})', 
             xy=(max_n, max_distance),
             xytext=(max_n-3, max_distance),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )


plt.title('Average distance between closest frequency pair')
plt.xlabel('Number of neighbours')
plt.ylabel('Distance (m)')
plt.grid(True)


plt.show()
############################################################################################

#############################################################################################
plt.plot(n, criticalD, 'b-o')


plt.title('Smallest frequency pair distance')
plt.xlabel('Number of neighbours')
plt.ylabel('Distance (m)')
plt.grid(True)


plt.show()




# Create the plot
plt.figure(figsize=(12, 6))

# Plot both lines
plt.plot(radius, radiusDistance, 'b-o', label='Average distance')
plt.plot(radius, minDistance, 'r-o', label='Minimum Distance')

# Customize the plot
plt.title('Average and mimimum distance vs minimun radius, N=5')
plt.xlabel('Radius (m)')
plt.ylabel('Distance (m)')
plt.grid(True)

# Add a legend
plt.legend()

# Show the plot
plt.show()

