class Strategy:
    def __init__(self, required_profit, required_pct_change_min, required_pct_change_max, required_volume):
        self.required_profit = required_profit
        self.required_pct_change_min = required_pct_change_min
        self.required_pct_change_max = required_pct_change_max
        self.required_volume = required_volume
