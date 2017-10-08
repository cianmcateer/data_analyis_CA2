import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import operator




############################################ DATA ##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
Type = ["medical","fashion","electrical","government","Property","funiture","educational","jewellery","jewellery","medical","fashion","fashion","fashion","fashion","medical","fashion","jewellery","government","government","electrical","educational","electrical","electrical","fashion","funiture","electrical","Property","electrical","fashion","fashion","fashion","electrical","funiture","medical","jewellery","educational","fashion","supermarket","jewellery","fashion","educational","fashion","supermarket","fashion","book","medical","supermarket","fashion","book","gaming","sports equipment","fashion","gaming","gaming","government","funiture","medical","electrical","educational","gaming","Property","book","fashion","fashion","jewellery","medical","government","Property","fashion","gaming","fashion","educational","Property","fashion","fashion","educational","sports equipment","gaming","educational","gaming","sports equipment","electrical","fashion","fashion","jewellery","Property","sports" "equipment","educational","medical","supermarket","book","electrical","Property","sports equipment","educational","electrical","fashion","fashion","fashion","educational"]

no_of_live_days = [502,189,76,518,423,234,155,543,1009,321,176,298,50,256,33,200,653,86,180,321,176,298,50,256,33,76,484,111,762,154,600,965,175,96,321,145,189,7,514,654,980,136,79,865,973,145,335,75,34,28,168,94,54,3,100000,264,433,42,354,87,522,1143,56,438,9,321,176,298,50,256,33,632,976,58,164,76,54,321,87,609,99,172,743,465,23,2,77,91,76,83,143,532,176,996,436,78,185,365,909,307]

down_time=[80.3,51.2,27.3,64.3,75.4,5.1,31.4,64.2,101.3,61.3,24.5,43.8,0.5,31.42,1.3,18,126.8,4.7,13.2,47.1,12.2,41.5,0.3,31.4,1.1,4.1,86.2,3.2,152.9,6.9,114,201.6,12,2.9,47.1,4.8,3.2,2.9,93.4,126.9,205.2,3.64,5.1,177.6,203.52,4.8,1020,3.2,0,0,10.32,0.6,1,0,56.1,33,73.9,0.8,54.9,10.7,95.28,244.32,0.3,75.12,9,47,12.24,41.52,8,31.44,0.6,121.7,204.24,5.3,10.8,5.7,3.4,23.1,2,116.2,5.3,11.3,148.3,81.6,3.3,0,34.1,4.5,31.6,21.6,4.3,91.2,8,212,75,9,14.4,57.6,188.2,43.7]

no_of_hits = [3000,12663,4921,34798,28341,11678,10890,56785,57609,21507,51792,29966,3350,17131,2221,13454,43789,6000
              ,12067,21507,11792,19966,350,7152,2211,4311,124280,7437,51054,10318,40200,54444,275,6431,21509,1009,5963,4489,34438
              ,43818,65660,9112,5293,57955,65191,8767,22441,5025,2278,1876,11256,6298,3618,201,567,17688,29011,2678,23576,5892,34974
              ,345,3752,29346,609,21507,11792,10099,3350,17152,2211,42344,76543,3886,10966,"NA",3651,21560,5878,40890,6632,11456,49879
              ,31145,1541,134,5159,6097,5092,5561,9581,356,11798,66732,29212,5226,12000,2456,"NA",3233]

still_alive= ["yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","no","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","no","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","no","yes","yes","yes","no","yes","yes","yes","yes"]

no_of_sales = [3960,886,1273,4176,3400,1401,762,8919,23043,5376,3625,1397,1340,2056,555,1608,420,720,4827,2580,2948,1397,42,858,265,1724.4,"NA",892,3573,1238,30150,6534,33,450,2581,403,1490,538.68,2410,32864,2000,3644,635,6954,7823,6575,1571,603,200,1350,756,6000,434,175,381,23000,21700,187,321,28929,0,4199,3700,7337,3522,"NA",0,156,402,"NA",884,296,0,272,1315,23,256,16170,4408,4907,798,1374,3481,3737,20003,0,619,427,1273,667,24,2,0,7689,7303,1306,840,172,8700,388]



average_sales_value = [15.67,19.99,398.1,1298.76,567.5,676.5,5.5,23.56,578.15,14.8,250.17,40.8,45.9,30.5,25.7,1056.89,75.5,50.9,77.5,156.9,300.1,98.5,567.14,23.6,256.75,276.75,0,176.5,23.84,124.5,673,1890,25000,32,75.9,15.99,324.7,45.67,129.99,98.1,150.99,67.5,76.5,43.6,23.56,78.15,14.8,250.17,30.8,45.9,1300.5,25.7,256.89,25.5,19.99,345,54.1,231,45,76.3,0,45.5,67.1,198.8,35.6,25.2,0,435,45,"NA",87.34,24,0,67,23.4,550,1345.87,33.3,23.4,92.1,499.99,56.5,67.8,34.2,234.4,0,155.42,78.9,50.5,141.3,54.67,4230.12,0,287,25,643,25,154.98,543.15,50]

average_user_age = [56,27,45,54,25,46,50,32,46,54,33,23,20,24,37,51,25,48,49,
                  33,43,47,53,19,31,26,43,23,22,22,38,43,34,19,33,39,41,35,22,24,25,28,56,65,46,
                  55,20,29,18,23,20,18,37,51,25,43,49,68,23,45,31,52,23,43,53,78,39,51,32,54,
                  23,38,58,31,21,22,27,43,42,53,23,33,23,20,19,37,51,25,43,49,33,43,42,53,43,56,183,19,31,5]


usability_rating = [1,2,2,4,1,2,3,3,4,4,2,4,3,2,4,2,2,2,3,3,2,1,2,1,2,3,3,2,4,1,2,
                    3,1,1,2,1,3,1,2,4,1,4,2,3,2,4,1,4,2,4,3,3,2,3,3,4,4,1,1,4,4,2,4,3,
                    4,3,2,1,4,3,2,1,4,3,1,2,4,4,1,3,3,3,3,3,2,4,1,2,3,3,2,3,1,4,2,2,1,2,3,1]

data = [Type,still_alive,no_of_sales,average_sales_value,average_user_age,usability_rating]

############################################ DATA ##################################################################################################################################################################################################################################################################################################################################################################################################################################################################


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#                                                             #
#                   UTILITY METHODS                           #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
returns list entered as arguement with outliers replaced by the list mean
"""
def replaceOutliers(l):
    mean = np.mean(l)
    std = np.std(l)

    for v in l:
        i = l.index(v)
        if not(v > mean - 2 * std) or not(v < mean + 2 * std):
            l.pop(i)
            l.insert(i,mean)
    return l
"""
returns mean of the list entered not including "NA"
"""
def getMean(l):
    total = 0
    for v in l:
        if v != "NA":
            total += v
    return total / len(l)

"""
Replaces elements with new values
"""
def replace(l,new,old): # Inserts mean where 'NA' exists
    for v in l:
        i = l.index(v)
        if v == old:
            l.pop(i)
            l.insert(i,new)
"""
Prints out list entered
"""
def printList(l):
    for v in l:
        print(v)

"""
Compares two lists side by side
"""
def compareLists(l1,l2):
    for i in range(len(l1)):
        print(l1[i],l2[i])

meanOfHits = getMean(no_of_hits)
meanOfSales = getMean(no_of_sales)
meanOfAverageSales = getMean(average_sales_value)
meanOfLiveDays = getMean(no_of_live_days)

replace(no_of_hits,meanOfHits,"NA")
replace(no_of_sales,meanOfSales,"NA")
replace(average_sales_value,meanOfAverageSales,"NA")
replace(Type,"sports equipment","sportsequipment")

def part1():

    print("/*******************************************************/")
    print("Part (i) Description of Data")
    print("/*******************************************************/")

    print("Handling missing data")
    print("Number of hits list has non applicable values to solve this we insert the mean value where 'NA' exists")

    def compare(l):

        yesList = []
        noList = []

        for i in range(len(still_alive)):
            if still_alive[i] is "yes":
                yesList.append(l[i])
            else:
                noList.append(l[i])
        print("yes mean",np.mean(yesList))
        print("no mean",np.mean(noList))

        print("yes mode",stats.mode(yesList)[0][0])
        print("No mode",stats.mode(noList)[0][0])

        print("Yes median",np.median(yesList))
        print("no median",np.median(noList))

        print("yes variance",np.var(yesList))
        print("no variance",np.var(noList))

        print("Yes standDev",np.std(yesList))
        print("No standDev",np.std(noList))


    print("Comparson number of live days of websites that are live or shut down")
    compare(no_of_live_days)
    print("")
    """
    Prints most common categories in ascending
    """
    def mostCommon():
        TypeSet = set(Type)
        typeList = list(TypeSet)
        print("CATEGORIES")
        printList(typeList)
        print("")
        medCount = 0
        fashionCount = 0
        electricalCount = 0
        governmentCount = 0
        propCount = 0
        bookCount = 0
        gamingCount = 0
        educationalCount = 0
        jewelleryCount = 0
        furnitureCount = 0
        supermarketCount = 0
        sportsEquipmentCount = 0

        rank = []

        for i in range(len(Type)):
            if Type[i] == "gaming":
                gamingCount += 1
            elif Type[i] == "educational":
                educationalCount += 1
            elif Type[i] == "fashion":
                fashionCount += 1
            elif Type[i] == "jewellery":
                jewelleryCount += 1
            elif Type[i] == "government":
                governmentCount += 1
            elif Type[i] == "medical":
                medCount += 1
            elif Type[i] == "supermarket":
                supermarketCount += 1
            elif Type[i] == "book":
                bookCount += 1
            elif Type[i] == "electrical":
                electricalCount += 1
            elif Type[i] == "sports equipment":
                sportsEquipmentCount += 1
            elif Type[i] == "Property":
                propCount += 1
            else:
                furnitureCount += 1

        rank.append(gamingCount)
        rank.append(educationalCount)
        rank.append(fashionCount)
        rank.append(jewelleryCount)
        rank.append(governmentCount)
        rank.append(medCount)
        rank.append(supermarketCount)
        rank.append(bookCount)
        rank.append(electricalCount)
        rank.append(sportsEquipmentCount)
        rank.append(propCount)
        rank.append(furnitureCount)
        dic = dict(zip(rank,typeList))
        sorted_d = sorted(dic.items(), key=operator.itemgetter(0))
        print("Least to most common Type of site")
        for k,v in sorted_d:
            print(v,"count",k)

    """
    Plots two lists against eachother (Scatter Plot)
    """
    def plotScatter(l1,l2):
        plt.scatter(l1)
        plt.scatter(l2)
        plt.show()

    """
    Plots two lists against eachother (Line Graph)
    """
    def plotList(l,message):
        plt.plot(l)
        plt.title(message)
        plt.show()

    plotList(no_of_live_days,"no. live days no outliers")
    live_days_without_outliers = replaceOutliers(no_of_live_days)
    plotList(live_days_without_outliers,"live days with outliers replaced")

    plotList(no_of_sales,"no. of sales")
    sales_without_outliers = replaceOutliers(no_of_sales)
    plotList(sales_without_outliers,"no. of sales without outliers")

    mostCommon()

def part2():

    print("/*******************************************************/")
    print("Part (ii) Descriptive Statistics")
    print("/*******************************************************/")

    """
    Compares the central tendencies and spreads of a numberic list that is still running
    """
    def compare(l):

        yesList = []
        noList = []

        for i in range(len(still_alive)):
            if still_alive[i] is "yes":
                yesList.append(l[i])
            else:
                noList.append(l[i])
        print("yes mean",np.mean(yesList))
        print("no mean",np.mean(noList))

        print("yes mode",stats.mode(yesList)[0][0])
        print("No mode",stats.mode(noList)[0][0])

        print("Yes median",np.median(yesList))
        print("no median",np.median(noList))

        print("yes variance",np.var(yesList))
        print("no variance",np.var(noList))

        print("Yes standDev",np.std(yesList))
        print("No standDev",np.std(noList))


    print("Comparson number of live days of websites that are live or shut down")
    compare(no_of_live_days)
    print("Comparson number of live days of hits")
    print("")
    compare(no_of_hits)
    print("")

    numbericData = [no_of_live_days,down_time,no_of_hits,no_of_sales,average_sales_value,average_user_age,usability_rating]
    listTitles = ["Number of Live Days","Down Time","Number of Hits","Number of Sales","Average Sales Values","Average User Age","Usability Rating"]

    def getAllMeans(data):
        means = [np.mean(d) for d in data]
        return means
    means = getAllMeans(numbericData)

    def getAllModes(data):
        modes = [stats.mode(d)[0][0] for d in data]
        return modes
    modes = getAllModes(numbericData)

    def getAllMedians(data):
        medians = [np.median(d) for d in data]
        return medians
    medians = getAllMedians(numbericData)

    def getAllVariance(data):
        variances = [np.var(d) for d in data]
        return variances
    variances = getAllVariance(numbericData)

    def getStandardDevs(data):
        standDevs = [math.sqrt(np.var(d)) for d in data]
        return standDevs
    standDevs = getStandardDevs(numbericData)

    def printLists(l1,l2):
        for i in range(len(l2)):
            print(l1[i] , str(l2[i]))

    print("Here are the central tendencies and speads of all our numberic data")
    print("")
    print("")
    print("MEAN")
    printLists(listTitles,getAllMeans(numbericData))
    print("")
    print("")
    print("MEDIAN")
    printLists(listTitles,getAllMedians(numbericData))
    print("")
    print("")
    print("MODE")
    printLists(listTitles,getAllModes(numbericData))
    print("")
    print("")
    print("VARIANCE")
    printLists(listTitles,getAllVariance(numbericData))
    print("")
    print("")
    print("STANDARD DEVIATION")
    printLists(listTitles,getStandardDevs(numbericData))

def part3():

    """
    Generates histogram of list entered and with or without outliers
    """
    def histogram(x,title,withOutlier):
        if withOutlier == True:
            x = replaceOutliers(x)


        plt.title(title)
        binBoundary = np.linspace(min(x),max(x),len(x))
        plt.hist(x,bins = binBoundary)
        plt.show()

    histogram(no_of_live_days,"No of live days",False)
    histogram(average_user_age,"average_user_age",False)
    histogram(down_time,"down_time",False)

    histogram(no_of_live_days,"No of live days (no outliers)",True)
    histogram(average_user_age,"average_user_age (no outliers)",True)
    histogram(down_time,"down_time (no outliers)",True)
    """
    Compares two lists as line graphs
    """
    def compareLists(x,y,xName,yName,title,withOutlier):

        if withOutlier == True:
            x = replaceOutliers(x)
            y = replaceOutliers(y)
            title += " with reduced outliers"

        plt.plot(x,label=xName, marker = "o")
        plt.title(title)
        plt.plot(y,marker="o")
        plt.show()

    compareLists(no_of_sales,average_sales_value,"no. of sales","average sale values","average_sales_value vs no_of_sales",False)
    compareLists(average_user_age,down_time,"average_user_age","down_time","average_user_age vs down_time",False)
    compareLists(usability_rating,no_of_hits,"average_user_age","down_time","average_user_age vs down_time",False)

    compareLists(no_of_sales,average_sales_value,"no. of sales","average sale values","average_sales_value vs no_of_sales",True)
    compareLists(average_user_age,down_time,"average_user_age","down_time","average_user_age vs down_time",True)
    compareLists(usability_rating,no_of_hits,"average_user_age","down_time","average_user_age vs down_time",True)

    """
    Scatter Plot comparison of two lists
    """
    def scatter(x,y,xLabel,yLabel,title,withOutlier):

        if withOutlier == True:
            x = replaceOutliers(x)
            y = replaceOutliers(y)
            title += " with reduced outliers"
        plt.scatter(x,y)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()

    scatter(no_of_hits,average_sales_value,"no. of sales","average sale values","no. hits vs average_sales_value",False)
    scatter(average_user_age,average_sales_value,"average_user_age","no_of_hits","average_user_age vs average_sales_value",False)
    scatter(usability_rating,down_time,"average_user_age","down_time","average_user_age vs down_time",False)

    scatter(no_of_hits,average_sales_value,"no. of sales","average sale values","no. hits vs average_sales_value",True)
    scatter(average_user_age,average_sales_value,"average_user_age","no_of_hits","average_user_age vs average_sales_value",True)
    scatter(usability_rating,down_time,"average_user_age","down_time","average_user_age vs down_time",True)


def part4():

    print("/*******************************************************/")
    print("Part (iv) Correlation & Simple Linear Regression")
    print("/*******************************************************/")
    """
    Calculates the pearson coeffcient of two lists
    """
    def getPearson(x,y):
        list1 = []
        list2 = []

        for i in x:
            # Sum of each value subtracted by mean divided by the standard deviation
            list1.append((i - np.mean(x)) / np.std(x))

        for i in y:
            # ditto
            list2.append((i - np.mean(y)) / np.std(y))

        return (sum([i*j for i,j in zip(list1,list2)])) / (len(x)-1)

    print("PEARSON CORRELATION")
    print("live days vs down time",getPearson(no_of_live_days,down_time))
    print("Average sale value vs average user age",getPearson(average_sales_value,average_user_age))
    print("No. of sales vs no. of hits",getPearson(no_of_sales,no_of_hits))
    """
    Draws scatter plot of lists with regression line drawn in
    """
    def scatter_regressionLine(x,y,xLine,yLine,title,withOutlier):

        if withOutlier == True:
            x = replaceOutliers(x)
            y = replaceOutliers(y)
            title += " (reduced outliers)"

        # Scatter plot
        plt.title(title)
        plt.scatter(x, y)
        plt.xlabel(xLine)
        plt.ylabel(yLine)

        # Add least squares regression line
        axes = plt.gca()
        m, b = np.polyfit(x, y, 1)
        X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
        # Add line to graph
        plt.plot(X_plot, m*X_plot + b, '-')
        plt.show()

    scatter_regressionLine(no_of_live_days,down_time,"live days","down time","live days vs down time",False)
    scatter_regressionLine(average_sales_value,average_user_age,"Average Sales Values","Average User Age","Average sale value vs average user age",False)
    scatter_regressionLine(no_of_sales,no_of_hits,"no. of sales","no of hits","No. of sales vs no. of hits",False)

    scatter_regressionLine(no_of_live_days,down_time,"live days","down time","live days vs down time",True)
    scatter_regressionLine(average_sales_value,average_user_age,"Average Sales Values","Average User Age","Average sale value vs average user age",True)
    scatter_regressionLine(no_of_sales,no_of_hits,"no. of sales","no of hits","No. of sales vs no. of hits",True)

    def squareList(x):
        return sum([i ** 2 for i in x])

    def multiplyList(a,b):
        return sum([x * y for x,y in zip(a,b)])

    def getB(x,y):
        countByxByY = len(x) * multiplyList(x,y)
        sumX = sum(x)
        sumY = sum(y)

        top = countByxByY - (sumX * sumY)

        countByxSquared = len(x) * squareList(x)
        sumXSquared = sumX ** 2

        bottom = countByxSquared - sumXSquared

        return float(top / bottom)

    def getA(x,y):
        sumY = sum(y)
        sumXSquared = squareList(x)
        sumX = sum(x)
        xTimesY = multiplyList(x,y)

        top = (sumY * sumXSquared) - (sumX * xTimesY)

        countByxSquared = len(x) * sumXSquared
        totalXSquared = sumX ** 2

        bottom = countByxSquared - totalXSquared

        return top / float(bottom)

    def getB(x,y):

        countByxByY = len(x) * multiplyList(x,y)
        sumX = sum(x)
        sumY = sum(y)

        top = countByxByY - (sumX * sumY)

        countByxSquared = len(x) * squareList(x)
        totalXSquared = sumX ** 2

        bottom = countByxSquared - totalXSquared

        return top / float(bottom)

    def getLinRegression(x,y,index):
        a = getA(x,y)
        b = getB(x,y)
        print(a)
        print(b)

        regression = (a + (b * x[index]))
        print(x[index],regression)


    getLinRegression(no_of_live_days,down_time,0)



def part5():

    print("/*******************************************************/")
    print("Part (v) Inferential Statistics")
    print("/*******************************************************/")
    """
    Obtains confidence interval of list along with entered z-score
    """
    def confidenceInterval(l,z_score):
        mean = np.mean(l)
        stdev = np.std(l)

        x = mean - z_score * (stdev / (math.sqrt(len(l))))
        y = mean + z_score * (stdev / (math.sqrt(len(l))))

        return [x,y]

    print("CONFIDENCE INTERVALS")
    print("usability_rating (68%)",confidenceInterval(usability_rating,1))
    print("Down time (68%)",confidenceInterval(down_time,1))
    print("No. of hits (68%)",confidenceInterval(no_of_hits,1))
    print("average_sales_value (68%)",confidenceInterval(average_sales_value,1))
    print("No of live days (68%)",confidenceInterval(no_of_live_days,1))
    print("")
    print("usability_rating (95%)",confidenceInterval(usability_rating,1.96))
    print("Down time (95%)",confidenceInterval(down_time,1.96))
    print("No. of hits (95%)",confidenceInterval(no_of_hits,1.96))
    print("average_sales_value (95%)",confidenceInterval(average_sales_value,1.96))
    print("No of live days (95%)",confidenceInterval(no_of_live_days,1.96))
    print("")
    print("usability_rating (99.7%)",confidenceInterval(usability_rating,2.58))
    print("Down time (99.7%)",confidenceInterval(down_time,2.58))
    print("No. of hits (99.7%)",confidenceInterval(no_of_hits,2.58))
    print("average_sales_value (99.7%)",confidenceInterval(average_sales_value,2.58))
    print("No of live days (99.7%)",confidenceInterval(no_of_live_days,2.58))



    print("HYPOTHESIS TESTING")
    print("HYPOTHESIS: There is a Correllation between usability rating and number of hits")
    print("ALTERNATIVE HYPOTHESIS: Websites that have  higher ratings also have a higher  level of hits")
    print("NULL HYPOTHESIS: They are independent of each other")
    """
    Returns list of highest hits and ratings
    """
    def hypothesisTest(l1,l2):

        print("TOP ten Hits and ratings")
        for i in range(len(l2)):
            if l2[i] > (np.mean(l2) + np.std(l2)):
                print(l1[i],l2[i])

    hypothesisTest(usability_rating,no_of_hits)

    print("Answer: Alternative Hypothesis is true")

def options():
    options = ["Exit","Description of Data","Descriptive Statistics","Plots","Correllation and linear regression","Hypothesis Testing"]
    for i in range(len(options)):
        print(str(i),options[i])
"""
Switch statement like menu
"""
def run():
    # Menu
    loop = 1
    while loop == 1:
        options()
        x = input("Choose from the above")
        if x == 1:
            part1()
        elif x == 2:
            part2()
        elif x == 3:
            part3()
        elif x == 4:
            part4()
        elif x == 5:
            part5()
        elif x == 0:
            loop = 0
            print("Goodbye")
        else:
            print("invalid input")
run()
