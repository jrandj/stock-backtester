# Stock-Backtester

Stock-backtester is a Python application for testing trading strategies against historical data. 
Running core.py will test strategies of buying and selling a stock based on technical indicators. Buy signals are 
generated using:
* Volume above 20 day moving average
* Close price change compared to previous day

Sell signals are generated using:
* Profit required from position

The minimum, maximum and step size for the strategy parameters is defined in the required_profit,
required_pct_change_max and required_volume arrays in config.py.

### Output
The output is stored in three SQL DB tables.

## Getting Started

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install stock-backtester. Using Windows Command Prompt:

1. Create a virtual environment
    ```bash
    py -m venv venv
    ```
2. Activate virtual environment
    ```bash
    "venv/Scripts/activate.bat"
    ```
3. Install dependencies
    ```bash
    pip3 install -r requirements.txt
    ```
### Prerequisites

#### Data



## Usage

Running

## License
[MIT](https://choosealicense.com/licenses/mit/)