"""
file: FIS-HW3.py
language: python3
description: FIS HW 3, regression by gradient descent
author: tpm6421@rit.edu (Trisha Malhotra)
"""
__author__ = "Trisha Malhotra"

def main():
    """
    Main function , takes data input
    into list, and calls the function: regression_run
    :return:pass
    """
    with open("hw3data.txt") as f:
        listdata = [line.split() for line in f]

    # using average as final value when [x-y,z]
    listdata[3]=['19','50.4']
    listdata[5] = ['20.5', '49.4']
    listdata[8] = ['37', '52.8']
    listdata[10] = ['47', '55.9']
    listdata[18] = ['36', '52.9']
    listdata[19] = ['43', '56.4']
    listdata[22] = ['39', '51.8']
    #converting to float
    newlistdata = [[(float(j)) for j in i] for i in listdata]
    print("Input data:")
    print(newlistdata)
    regression_run(newlistdata)


def regression_run(listdata):
    """
    Analyses the data , and calls helper functions
    :param listdata: input data list
    :return: pass
    """
    points = listdata
    learning_rate = 0.001
    starting_b = 0 # w0
    starting_m = 0 # w1
    starting_c = 0 # w2
    iteration_num = 35
    print ("\nRunning gradient descent using linear model...")
    [b, m] = run_linear(points, starting_b, starting_m, learning_rate, iteration_num)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(iteration_num, b, m, compute_error_linear(b, m, points)))

    print("\nRunning gradient descent using quadratic model ...")
    [b, m, c] = run_quadratic(points, starting_b, starting_m, starting_c, learning_rate, iteration_num)
    print("After {0} iterations b = {1}, m = {2}, c = {3}, error = {4}".format(iteration_num, b, m, c,
                                                                               compute_error_quadratic(b, m, c, points)))
    print("\nQuadratic model turns a linear model into a curve, \nwhile the error is higher that that of the linear model.")


def run_linear(points, starting_b, starting_m, learning_rate, iteration_num):
    """
    This function calls step_gradient 
    :param points: list
    :param starting_b: w0
    :param starting_m: w1
    :param learning_rate: w2 
    :param iteration_num: 35
    :return: [b, m]
    """
    b = starting_b
    m = starting_m
    for i in range(iteration_num):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]


def run_quadratic(points, starting_b, starting_m, starting_c, learning_rate, iteration_num):
    """
    Runs the regression using quadratic model
    calls function quadratic_gradient
    :param points: list
    :param starting_b: w0
    :param starting_m: w1
    :param starting_c: w2
    :param learning_rate: 0.001 
    :param iteration_num: 35
    :return: 
    """
    b = starting_b
    m = starting_m
    c = starting_c
    for i in range(iteration_num):
        b, m , c = quadratic_gradient(b, m, c, points, learning_rate)
    return [b, m , c]


def quadratic_gradient(b_current, m_current,c_current, points, learning_rate):
    """
    Calculates new b, m c
    :param b_current: w0
    :param m_current: w1
    :param c_current: w2
    :param points: list
    :param learning_rate:0.001
    :return:[newb, newm, newc]
    """
    b_gradient = 0
    m_gradient = 0
    c_gradient = 0

    for i in range(0, len(points)):
        x = points[i][0]
        new_x = float(x)
        y = points[i][1]
        new_y = float(y)
        b_gradient += - 2 * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
        m_gradient += - 2 * new_x * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
        c_gradient += - 2 * ((new_x)**2) * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    new_c = c_current - (learning_rate * m_gradient)
    return [new_b, new_m, new_c]


def step_gradient(b_current, m_current, points, learning_rate):
    """
    calculates new b, m
    :param b_current: w0
    :param m_current: w1
    :param points: list
    :param learning_rate:0.001
    :return: [newb, newm]
    """
    b_gradient = 0
    m_gradient = 0

    for i in range(0, len(points)):
        x = points[i][0]
        new_x = float(x)
        y = points[i][1]
        new_y = float(y)
        b_gradient += - 2 * (new_y - ((m_current * new_x) + b_current))
        m_gradient += - 2 * new_x * (new_y - ((m_current * new_x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def compute_error_linear(b, m, points):
    """
    Calculates error for linear model
    :param b: w0
    :param m: w1
    :param points: list
    :return: total error
    """
    total_error = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        total_error += (y - (m * x + b)) ** 2
    return total_error

def compute_error_quadratic(b, m, c, points):
    """
    calculates error for quadratic model
    :param b: w0
    :param m: w1
    :param c: w2
    :param points:list
    :return: total error
    """
    total_error = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        total_error += (y - ((c * (x**2)) + m * x + b)) ** 2
    return total_error


def piecewise_linear(b_current, m_current, points, learning_rate):
    """
    Calculates new b, m c
    :param b_current: w0
    :param m_current: w1
    :param points: list
    :param learning_rate:0.001
    :return:[newb, newm]
    """
    b_gradient = 0
    m_gradient = 0
    tmiddle = 50
    # generating middle temperature:
    for threshold in points:
        t= points[threshold]
        if t > tmiddle:
            t+= 1
        else:
            newthreshold = t
    #first half
    for i in range(0,(points[newthreshold])):
        x = points[i][0]
        new_x = float(x)
        y = points[i][1]
        new_y = float(y)
        b_gradient += - 2 * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
        m_gradient += - 2 * new_x * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
        c_gradient += - 2 * ((new_x)**2) * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    new_c = c_current - (learning_rate * m_gradient)
    return [new_b, new_m, new_c]

    # second half
    for i in range((points[newthreshold+1],0)):
        x = points[i][0]
        new_x = float(x)
        y = points[i][1]
        new_y = float(y)
        b_gradient += - 2 * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
        m_gradient += - 2 * new_x * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
        c_gradient += - 2 * ((new_x)**2) * (new_y - (c_current * (new_x**2) + (m_current * new_x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    new_c = c_current - (learning_rate * m_gradient)
    return [new_b, new_m, new_c]

if __name__== '__main__':
    main()