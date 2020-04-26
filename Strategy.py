class Strategy:
    """
    A class used to represent a Strategy.

    Attributes
    ----------
    required_profit : numeric
        The profit multiple required to exit a position.
    required_pct_change_min : numeric
        The price change lower bound to generate a buy signal.
    required_pct_change_max : numeric
        The price change upper bound to generate a buy signal.
    required_volume : numeric
        The required multiple of the 20D MA volume to generate a buy signal.

    Methods
    -------

    """
    def __init__(self, required_profit, required_pct_change_min, required_pct_change_max, required_volume):
        """
        Parameters
        ----------
        required_profit : numeric
            The profit multiple required to exit a position.
        required_pct_change_min : numeric
            The price change lower bound to generate a buy signal.
        required_pct_change_max : numeric
            The price change upper bound to generate a buy signal.
        required_volume : numeric
            The required multiple of the 20D MA volume to generate a buy signal.

        """

        self.required_profit = required_profit
        self.required_pct_change_min = required_pct_change_min
        self.required_pct_change_max = required_pct_change_max
        self.required_volume = required_volume
