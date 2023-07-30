import numpy as np
import pandas as pd
from empyrical import sharpe_ratio
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def stock_close_prices(key):
    '''return a list containing stock close prices from a .csv file'''
    prices = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        prices.append(float(line.split(",")[4]))
    return prices


def generate_price_state(stock_prices, end_index, window_size):
    '''
    return a state representation, defined as
    the adjacent stock price differences after sigmoid function (for the past window_size days up to end_date)
    note that a state has length window_size, a period has length window_size+1
    '''
    start_index = end_index - window_size
    if start_index >= 0:
        period = stock_prices[start_index:end_index+1]
    else: # if end_index cannot suffice window_size, pad with prices on start_index
        period = -start_index * [stock_prices[0]] + stock_prices[0:end_index+1]
    return sigmoid(np.diff(period))


def generate_portfolio_state(stock_price, balance, num_holding):
    '''logarithmic values of stock price, portfolio balance, and number of holding stocks'''
    return [np.log(stock_price), np.log(balance), np.log(num_holding + 1e-6)]


def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
    '''
    return a state representation, defined as
    adjacent stock prices differences after sigmoid function (for the past window_size days up to end_date) plus
    logarithmic values of stock price at end_date, portfolio balance, and number of holding stocks
    '''
    prince_state = generate_price_state(stock_prices, end_index, window_size)
    portfolio_state = generate_portfolio_state(stock_prices[end_index], balance, num_holding)
    return np.array([np.concatenate((prince_state, portfolio_state), axis=None)])


def treasury_bond_daily_return_rate():
    r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
    return (1 + r_year)**(1 / 365) - 1


def maximum_drawdown(portfolio_values):
    end_index = np.argmax(np.maximum.accumulate(portfolio_values) - portfolio_values)
    if end_index == 0:
        return 0
    beginning_iudex = np.argmax(portfolio_values[:end_index])
    return (portfolio_values[end_index] - portfolio_values[beginning_iudex]) / portfolio_values[beginning_iudex]
