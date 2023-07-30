import numpy as np

class Engine:
    """Defines the environment function from the generator engine.
       Expects the following:
        - reset() to reset the env a start position(s)
        - step() to make an action and update the game state
        - legal_moves_generator() to generate the list of legal moves
    """
    def __init__(self,
                 stock_name:str='DJI_2010-2018',
                 trading_days:int=365,
                 window_size:int=5,
                 balance:int=50000) -> None:
        """Initialize Engine"""
        self.stock_name = stock_name
        self.stock_prices = Engine.stock_close_prices(stock_name)
        # --- 
        self.trading_days = trading_days # number of trading days
        self.t = 0  # current trading day
        if self.t>len(self.stock_prices):
            self.t = len(self.stock_prices)-1
        self.initial_portfolio_value = balance # initial portfolio value
        self.balance = balance # current portfolio value
        self.window_size = window_size # window size for state representation
        self.inventory = [] # stocks in hand
        self.return_rates = [] # daily return rates
        self.portfolio_values = [balance] # portfolio values
        self.buy_dates = [] # buy dates
        self.sell_dates = [] # sell dates
        self.reward = 0 # reward
        print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        print("Engine initialized - Stock: ", stock_name)
 
    # ------------------------------------------------------------
    # STATIC METHODS FOR ENGINE MDP
    # Get stock prices from a .csv file
    @staticmethod
    def stock_close_prices(key:str):
        '''return a list containing stock close prices from a .csv file'''
        prices = []
        lines = open("./environment/data/" + key + ".csv", "r").read().splitlines()
        for line in lines[1:]:
            prices.append(float(line.split(",")[4]))
        return prices
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
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
        return Engine.sigmoid(np.diff(period))

    @staticmethod
    def generate_portfolio_state(stock_price, balance, num_holding):
        '''logarithmic values of stock price, portfolio balance, and number of holding stocks'''
        return [np.log(stock_price), np.log(balance), np.log(num_holding + 1e-6)]

    @staticmethod
    def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
        '''
        return a state representation, defined as
        adjacent stock prices differences after sigmoid function (for the past window_size days up to end_date) plus
        logarithmic values of stock price at end_date, portfolio balance, and number of holding stocks
        '''
        prince_state = Engine.generate_price_state(stock_prices, end_index, window_size)
        portfolio_state = Engine.generate_portfolio_state(stock_prices[end_index], balance, num_holding)
        return np.array([np.concatenate((prince_state, portfolio_state), axis=None)])[0]

    @staticmethod
    def treasury_bond_daily_return_rate():
        r_year = 2.75 / 100  # approximate annual U.S. Treasury bond return rate
        return (1 + r_year)**(1 / 365) - 1
    # ------------------------------------------------------------
    
    def hold(self):
        # Do nothing
        return 'Hold'
        
    def buy(self):
        if self.balance > self.stock_prices[self.t]:
            self.balance -= self.stock_prices[self.t]
            self.inventory.append(self.stock_prices[self.t])
            return 'Buy: ${:.2f}'.format(self.stock_prices[self.t])
        
    def sell(self):
        if len(self.inventory) > 0:
            self.balance += self.stock_prices[self.t]
            bought_price = self.inventory.pop(0)
            profit = self.stock_prices[self.t] - bought_price
            return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(self.stock_prices[self.t], profit)
        
    def reset(self):
        """Fully reset the environment."""
        #obs, _ = self.Environment.reset()
        self.t = 0
        self.balance = self.initial_portfolio_value
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [self.initial_portfolio_value]
        self.reward = 0
        obs = Engine.generate_combined_state(0, self.window_size, self.stock_prices, self.balance, len(self.inventory))
        return obs

    
    def step(self, state:any, action:any):
        """Enact an action."""
        #obs, reward, terminated = self.Environment.step(action)
        # In problems where the agent can choose to reset the env
        if (state=="ENV_RESET")|(action=="ENV_RESET"):
            self.reset()
        
        # ---
        # ENACT ACTION
        self.t+=1

        if action == 0: # hold
            execution_result = self.hold()
        if action == 1: # buy
            execution_result = self.buy()      
        if action == 2: # sell
            execution_result = self.sell()    
        # ---

        # UPDATE STATE
        # Execution adds stock to inventory for next state
        obs = Engine.generate_combined_state(0, self.window_size, self.stock_prices, self.balance, len(self.inventory))
        # ---

        # REWARD
        # check execution result
        if execution_result=='Hold':
            self.reward -= Engine.treasury_bond_daily_return_rate() * self.balance  # missing opportunity

        # calculate reward
        current_portfolio_value = len(self.inventory) * self.stock_prices[self.t] + self.balance
        unrealized_profit = current_portfolio_value - self.initial_portfolio_value
        self.reward += unrealized_profit
        # ---

        # Episode end conditions    
        if self.t == self.trading_days:
            terminated = True
        elif self.balance <=0:
            terminated = True
        else:
            terminated = False
        print("\n - ", self.balance)
        return obs, self.reward, terminated

    def legal_move_generator(self, obs:any=None):
        """Define legal moves at each position"""
        legal_moves = [0,1,2] # 'hold', 'buy', 'sell'
        return legal_moves

