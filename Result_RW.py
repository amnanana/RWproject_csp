#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 21:01:51 2020

@author: guoshuping
"""



import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import math
from timeit import default_timer as timer
from copy import deepcopy


class RoofDetails(object):
    def __init__(self):
        pass
    
    #the details of each type of conservatory
    def roofType(self,structure):
        if structure == '3x3':
            roof_dict = {'hip':[2275,2275],'ridge':[1500],
                         'rafter':[1698,1698,1665,847,847,1665,847,847,1665,1698,1698]}
        if structure == '3.3x3':
            roof_dict = {'hip':[2404,2404],'ridge':[1500],
                         'rafter':[1864,1864,1830,998,927,1696,927,998,1830,1864,1864]}
        if structure == '4.6x4.2':
            roof_dict = {'hip':[2275,2275],'ridge':[1500],
                         'rafter':[1698,1698,1665,913,913,1665,913,913,1665,1698,1698]}
        if structure == '3.6x2.7':
            roof_dict = {'hip':[2722,2722],'ridge':[900],
                         'rafter':[2029,2029,1996,1078,1078,1996,1012,1012,1996,2029,2029]}
        if structure == '2.9x3.5':
            roof_dict = {'hip':[2419,2419],'ridge':[1750],
                         'rafter':[1643,1643,1610,879,1027,1885,1027,879,1610,1643,1643]}
            
        return roof_dict
    
    #generate bars from specific conservatory type
    def generateBars(self,roofList,barType):
        barList = []
        for roof in roofList:
            roof1 = self.roofType(roof)
            bars = roof1[barType]
            for bar in bars:
                barList.append(bar)
        return barList
    
    #generate roof orders according to given probability
    def generate_orders_with_probability(self,orderNumber):
        orderProba = {'3.3x3': 0.3, '3x3': 0.2, '3.6x2.7': 0.2, '2.9x3.5': 0.2, '4.6x4.2': 0.1}
        roofs = []
        for roof,proba in orderProba.items():
            if proba != min(orderProba.values()):
                roofNum = round(orderNumber*proba)
            else:
                roofNum = orderNumber - len(roofs)
            for i in range(roofNum):
                roofs.append(roof)
        return roofs
    
    #add more details of each roof member    
    def get_details(self,roofList,barType):
        
        if barType == 'ridge':
            Bars = self.generateBars(roofList,'ridge')
            stockSize = [3000,5000]
            stockPrice = [15,24]
            waste = 350
        if barType == 'rafter':
            Bars = self.generateBars(roofList,'rafter')
            stockSize = [6000,7000]
            stockPrice = [20,23]
            waste = 1000
        if barType == 'hip':
            Bars = self.generateBars(roofList,'hip')
            stockSize = [6000,7000]
            stockPrice = [25,29]
            waste = 1000
        return Bars, stockSize,stockPrice, waste
    
    

class Sample():
    
    def __init__(self):
        pass


    #generate sample by cutting bars sequentially after shuffling the order list 
    def generateSample(self,sL,bL,pL):
        #time_start = timer()
        newBars = bL.copy()
    
        currentLeft,currentStock = [],[]
        samplePattern = []

        while newBars:
            choiceStock = random.choice(sL)
            res,tmpSum,leftover = [],0,choiceStock
            
            while tmpSum <= choiceStock and newBars:
                if leftover >= newBars[0]:
                    currentBar = newBars.pop(0)
                    res.append(currentBar)
                    tmpSum += currentBar
                    leftover -= currentBar
    
                else:
                    break
            if choiceStock == min(sL):
                stockPrice = min(pL)
            else:
                stockPrice = max(pL)
            samplePattern.append([choiceStock,stockPrice,res,leftover])
        
        return samplePattern
    
    #calculate the objective function by assigning specific weight for each decision variables
    def getObjective(self,result):
        total_cost,total_leftover =0,0
        
        for i in range(len(result)):
            total_cost += result[i][1]
            total_leftover += result[i][-1]
        
        obj = 0.4*len(result) + 0.3* total_cost + 0.3* total_leftover
        
        return obj

#algorithm 1: the best fit decreasing
class bestFitDecreasing(Sample):
    def __init__(self):
        super().__init__()
        
    def BFD(self,barL,sL,pL):
        bL = barL.copy()
        bL.sort()
        shortStock,longStock = min(sL),max(sL)
        shortStockPrice,longStockPrice = min(pL),max(pL)

        optPattern = []
        while bL:
            #for the last demand, directly adding a standard bar
            if len(bL) == 1:
                if bL[0]>shortStock:
                    remain = longStock - bL[0]
                    optPattern.append([longStock,longStockPrice,[bL[0]],remain])
                    break
                else:
                    remain = shortStock - bL[0]
                    optPattern.append([shortStock,shortStockPrice,[bL[0]],remain])
                    break
                
            firstRes = []
            for stock in sL:
                res = []    #to keep optimal plan
                shortRoll = min(bL)
                tmpSum = 0
                leftover = 0
                i = len(bL)-1
                while tmpSum <= stock and bL:
                    #if standard bar could be cut - add plan - delete demand
                    tmpSum += bL[i]
                    if tmpSum <= stock and i>=0:
                        res.append(bL[i])
                        i -= 1
                        leftover = stock - tmpSum
    
                    elif leftover >= shortRoll and i>=0:
                        #if no - find other suitable bar size
                        tmpSum -= bL[i]
                        n=0
                        while bL[n] <= leftover:
                            n += 1
                        
                        bestLeft = 0
                        for t in range(n):
                            if leftover > 0:
                                curRoll = bL[t]
                                curLeft = leftover - curRoll
                                
                                if curLeft > bL[t+1]:
                                    for y in range(1,n):
                                        newRoll = bL[y]
                                        newLeft = newRoll + curRoll
                                        if newLeft >= bestLeft and newLeft <= leftover:
                                            if newLeft == leftover:
                                                res.append(bL[t])
                                                res.append(bL[y])
                                                tmpSum += newLeft
                                                i -= 2
                                                leftover = 0
                                                break
                                            else:
                                                
                                                bestLeftIndex = []
                                                bestLeftIndex.append(bL[t])
                                                bestLeftIndex.append(bL[y])
                                                bestLeft = newLeft
                                elif curRoll > bestLeft:
                                    bestLeftIndex = []
                                    bestLeftIndex.append(bL[t])
                                    bestLeft = curRoll
                                
                      
                        for value in bestLeftIndex:
                            tmpSum += value
                            res.append(value)
                            i -= 1
                            leftover = stock - tmpSum
                    
                    else:
                        # if above situations couldn't meet - add another bar to cut
                        break
                            
                #compare the efficiency of different stock sizes
                if stock == shortStock:
                    price = shortStockPrice
                else:
                    price = longStockPrice
          
                    
                if len(firstRes) == 0:
                    firstRes += [stock,price,res,leftover]#index for values: res-0; leftover-1
                    
                else:
                    #compare which size of stock bar could cut more bars
                    if len(res) > len(firstRes[-2]):
                        optPattern.append([stock,price,res,leftover])
                    elif len(res) == len(firstRes[-2]):
                        if leftover < firstRes[-1]:
                            optPattern.append([stock,price,res,leftover])
                        else:
                            optPattern.append(firstRes)
                    else:
                        optPattern.append(firstRes)
            chosenBars = optPattern[-1][-2]
            
            for b in chosenBars:
                bL.remove(b)

        return optPattern

    #iterate the algorithm for 500 times
    def getBFD(self,barL,sL,pL,barType,seed,algorithm): 
        bfdRes = []
        i=0
        
        while i < iterations:
            time_start = timer()
            
            res = self.BFD(barL,sL,pL)
            
            time_end = timer()
            time_difference = time_end - time_start
            obj = super().getObjective(res)
            bfdRes.append([barType,seed,algorithm,i,obj,time_difference,res])
            i += 1
            
        return bfdRes


#algorithm 2: the random search
class RandomSearch(Sample):
    def __init__(self):
        super().__init__()



    def getRS(self,bL,sL,pL,barType,seed,algorithm):
        rsRes = []
        i=0
        opt = super().generateSample(sL,bL,pL)
        
        while i < iterations:
            time_start = timer()
            sample_obj = super().getObjective(opt)
            random.shuffle(bL) #shuffle the order list and call the sample function again to search new solution
            rs = super().generateSample(sL,bL,pL)
            rs_obj = super().getObjective(rs)
            
            if rs_obj < sample_obj: # iterative aspect of function
                opt = rs
                
            time_end = timer()
            time_difference = time_end - time_start
            currentObj = super().getObjective(opt)
            rsRes.append([barType,seed,algorithm,i,currentObj,time_difference,opt])
            i += 1
            
        return rsRes
    
    

class SimulatedAnnealing(Sample):
    def __init__(self):
        super().__init__()


    
    #NM2: random chose one bar to another cutting pattern and change stocksize and price
    def NM(self,result,sL,pL):
        res = deepcopy(result)
        #Move a bar to an arbitrary position in another cutting pattern.    
        bar_amount = len(res)
        stock_diff = max(sL)-min(sL)
        index_list = list(range(0,bar_amount))
        i = 0

        bar_index = np.random.choice(index_list)
            
        index_list.remove(bar_index)    
        while index_list:
        # Generate a random index for the bar to be swapped
            
            select_pattern = res[bar_index][2]
            select_bar = np.random.choice(select_pattern)
            
            new_bar_index = np.random.choice(index_list)
            #move select bar if possible
            new_pattern = res[new_bar_index][2]
            new_bar = np.random.choice(new_pattern)
            
            if len(select_pattern) == 1:
                if select_bar <= res[new_bar_index][3]:
                
                    res[new_bar_index][2].append(select_bar)
                    res[new_bar_index][3]= res[new_bar_index][0] - sum(res[new_bar_index][2])
                    del res[bar_index]
                    return res
                    
                elif res[new_bar_index][0] == min(sL) and sum(res[new_bar_index][2]) + select_bar <= max(sL):
                    res[new_bar_index][2].append(select_bar)
                    new_left = max(sL) - sum(res[new_bar_index][2])
                    res[new_bar_index] = [max(sL),max(pL),res[new_bar_index][2],new_left]
                
                    del res[bar_index]
                    return res
                
                else:
                    index_list.append(bar_index)
                    bar_index = np.random.choice(index_list)
                    index_list.remove(bar_index) 
                    
            elif len(new_pattern) == 1:
                if new_bar <= res[bar_index][3]:
                    res[bar_index][2].append(new_bar)
                    
                    #update the leftover value of new cutting pattern
                    res[bar_index][3]= res[bar_index][0] - sum(res[bar_index][2])
                    del res[new_bar_index]
                    return res
                    
                elif res[bar_index][0] == min(sL) and sum(select_pattern) + new_bar <= max(sL):
                    res[bar_index][2].append(new_bar)
                    select_left = max(sL) - sum(res[bar_index][2])
                    res[bar_index] = [max(sL),max(pL),res[bar_index][2],select_left]
                
                    del res[new_bar_index]
                    return res
                
                else:
                    index_list.append(bar_index)
                    bar_index = np.random.choice(index_list)
                    index_list.remove(bar_index)
                
            elif select_bar <= res[new_bar_index][3]:
                
                res[new_bar_index][2].append(select_bar)
                #update the leftover value of new cutting pattern
                res[new_bar_index][3]= res[new_bar_index][0] - sum(res[new_bar_index][2])
                
                
                #remove selected bar from original cutting pattern
                res[bar_index][2].remove(select_bar)
                res[bar_index][3] = res[bar_index][0] - sum(res[bar_index][2])
                    
                return res
                
                
            #if cannot move select bar, than swap bars of two cutting patterns        
            else:
                
                barDiff = select_bar - new_bar
                
                if res[bar_index][3] + barDiff >= 0 and res[new_bar_index][3]- barDiff >=0:
                    
                    res[bar_index][2].append(new_bar)
                    res[new_bar_index][2].append(select_bar)
                    
                    
                    res[bar_index][2].remove(select_bar)
                    res[new_bar_index][2].remove(new_bar)
                    
                    
                    res[bar_index][3] = res[bar_index][0] - sum(res[bar_index][2])
                    res[new_bar_index][3] = res[new_bar_index][0] - sum(res[new_bar_index][2])
                    
                    
                    index_list.append(bar_index)
                    bar_index = np.random.choice(index_list)
                    index_list.remove(bar_index)
                    #the stop criterion, happened when the times of loop over the length of barList
                    i += 1
                    if i>bar_amount:
                        return res
                    
                else:
                    index_list.append(bar_index)
                    bar_index = np.random.choice(index_list)
                    index_list.remove(bar_index)
                    i += 1
                    if i>bar_amount:
                        return res
                    
            
                
          
            
    def getSA(self,bL,sL,pL,barType,seed,algorithm,decay_amount):
        T = 1
        decay = decay_amount
        saRes = []
        s = super().generateSample(sL,bL,pL)
        i=0
        
        while i < iterations:
            time_start = timer()
            
            # calculate y
            # generate a new s in the neighboorhood of s by transform function
            # some neighbouring state s′
            sNew = self.NM(s,sL,pL)
            yNew = super().getObjective(sNew)
            yOld = super().getObjective(s)
            difference = yNew - yOld
            if difference < 0 or np.random.random(1) < math.exp((-difference)/(yOld*T)):
                s = sNew
            time_end = timer()
            time_difference = time_end - time_start
            currentObj = super().getObjective(s)
            saRes.append([barType,seed,algorithm,i,currentObj,time_difference,s])
            i +=1
            T *= decay
            
            
            
        return saRes 




class Results(RoofDetails,bestFitDecreasing,RandomSearch,SimulatedAnnealing):
    def __init__(self):
        super().__init__()

    #run all algorithms with 500 iterations and 20 seeds under one order size
    def runAlgorithms(self,roofList,seedAmount,decay):

        allResult = []
        
        for se in range(seedAmount):
            random.seed(se)
            random.shuffle(roofList)
            for barName in roofBars:
                barList,stockList,stockPrice,wasteLimit = super().get_details(roofList,barName)
                
                for algorithm in algorithms:
                    if algorithm == 'Best Fit Decreasing':
                        res = super().getBFD(barList,stockList,stockPrice,barName,se,algorithm)
                        print(barName + ' seed ' + str(se) + ' BFD finished')
                        
                    elif algorithm == 'Random Search':
                        res = super().getRS(barList,stockList,stockPrice,barName,se,algorithm)
                        print(barName + ' seed ' + str(se) + ' RS finished')
                                 
                    else:
                        res= super().getSA(barList,stockList,stockPrice,barName,se,algorithm,decay)
                        print(barName + ' seed ' + str(se) + ' SA finished')
                        
                    allResult += res
        df = pd.DataFrame(allResult, columns=['BarType','Seed', 'Algorithm','Iteration','Objective','Iteration_sec','Results'])                 
        return df

    #separete the orders into 5 different order scales and conbime them to one for comparison
    def separateOrders(self,roofList,seedAmount,decay):

        separateRes = []
        l = len(roofList)
        #s = 0
        for se in range(seedAmount):
            random.seed(se)
            random.shuffle(roofList)
            for barName in roofBars:
                for n in range(1,11):
                    if l%n == 0:
                        orderAmount = int(l/n)
                        sliceRoof = [roofList[i:i+orderAmount] for i in range(0, len(roofList), orderAmount)]

                        for algorithm in algorithms:
                            c,combineRes,operateTime = 0,{},[]
                            for y in range(n):
                                subBarList,subStockList,subStockPrice,subWaste = super().get_details(sliceRoof[y],barName)
                                
                                if algorithm == 'Best Fit Decreasing':
                                    subRes = super().getBFD(subBarList,subStockList,subStockPrice,barName,se,algorithm)
                                elif algorithm == 'Random Search':
                                    subRes = super().getRS(subBarList,subStockList,subStockPrice,barName,se,algorithm)

                                else:
                                    subRes = super().getSA(subBarList,subStockList,subStockPrice,barName,se,algorithm,decay)

                                subTime = 0
                                for time in range(len(subRes)):
                                    subTime += subRes[time][-2]
                                operateTime.append(subTime)
                                res = subRes[-1][-1]
                                for x in range(len(res)):
                                    combineRes[c] = res[x]
                                    c += 1
                                    
                            print('=================='+algorithm +'====='+ str(y) +'====================')
                            combineObj = super().getObjective(combineRes)
                            totalTime = sum(operateTime)
                            separateRes.append([barName,se,algorithm,str(orderAmount)+'x'+str(n),combineObj,totalTime,combineRes])
                        print(barName + ' Seed' + str(se) +' separate finished')
                    
                else:
                    continue
        df= pd.DataFrame(separateRes,columns=['BarType','Seed', 'Algorithm','OrderScale','Objective','Time_sec','Results'])
        return df

    #run algorithms for the order amount from 10 to 100 by increasing 10 orders each time 
    def ordersResult(self,seedAmount,decay):
        ordersRes = pd.DataFrame(columns=['OrderAmount','BarType','Seed', 'Algorithm','Objective','Time_sec'])
        x = 0
        for orders in range(0,101,10):
            if orders == 0:
                continue
            else:
                curRoofList = super().generate_orders_with_probability(orders)
                curRes = self.runAlgorithms(curRoofList,seedAmount,decay)
                for barName in roofBars:
                    for se in range(seedAmount):
                        for algorithm in algorithms:
                            condition = curRes[(curRes['BarType']==barName) & (curRes['Seed']==se) & (curRes['Algorithm']==algorithm)]
                            time = condition['Iteration_sec'].sum()
                            optObj = condition['Objective'].min()
                            ordersRes.loc[x] = [orders,barName,se,algorithm,optObj,time]
                            x += 1
            print(str(orders)+ ' orders ' + ' finished')
        return ordersRes
                            
                
#global variables
roofNumber = 100
seedAmount = 20
decay = 0.95
iterations = 500
roofBars = ['ridge','hip','rafter']
algorithms = ['Best Fit Decreasing','Random Search','Simulated Annealing']

#class instance
Roof = RoofDetails()
roofList = Roof.generate_orders_with_probability(roofNumber)

R = Results()

run = R.runAlgorithms(roofList,seedAmount,decay)
run.to_csv('all_algorithms_Results_RW.csv')
print('=========================runAlgorithms done========================') 
print(run)

orderR = R.ordersResult(seedAmount,decay)
orderR.to_csv('orders_Results_RW.csv') 
print('=========================ordersResult done========================') 
print(orderR)

sep = R.separateOrders(roofList,seedAmount,decay)
sep.to_csv('separate_orders_Results_RW.csv') 
print('=========================separateOrders done========================') 
print(sep)
