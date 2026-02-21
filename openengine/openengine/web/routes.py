import json
import os
import re
import importlib
import importlib.util
import traceback
from datetime import datetime

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash

from openengine.data.yahoo_connector import YahooFinanceConnector
from openengine.strategies.sample_strategy import SampleStrategy
from openengine.strategies.base_strategy import BaseStrategy
from openengine.engine.backtester import Backtester
from openengine.engine.models import BacktestConfig, SizingMode, OOSConfig
from openengine.engine.oos_engine import run_out_of_sample_backtest
from openengine.engine.monte_carlo_engine import get_trade_returns, run_monte_carlo

main_bp = Blueprint('main', __name__)

# In-memory store for backtest results (simple approach; no DB needed)
_backtest_results = {}
_backtest_counter = 0

# Path to the strategies folder
_STRATEGIES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')

# Default template code for new strategies
_DEFAULT_STRATEGY_CODE = '''import pandas as pd
from openengine.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        """
        Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns [Open, High, Low, Close, Volume]

        Returns:
            pandas Series with values:
                1  = BUY signal
               -1  = SELL signal
                0  = HOLD (no action)
        """
        signals = pd.Series(0, index=data.index)

        # Example: Simple SMA crossover
        short_window = 10
        long_window = 30

        short_ma = data["Close"].rolling(window=short_window).mean()
        long_ma = data["Close"].rolling(window=long_window).mean()

        signals[short_ma > long_ma] = 1   # BUY
        signals[short_ma <= long_ma] = -1  # SELL

        return signals
'''


def _scan_user_strategies():
    """Scan the strategies folder for user-created .py files and return their metadata."""
    user_strategies = []
    if not os.path.exists(_STRATEGIES_DIR):
        return user_strategies

    # Files to skip (built-in)
    skip_files = {'__init__.py', 'base_strategy.py', 'sample_strategy.py', '__pycache__'}

    for filename in sorted(os.listdir(_STRATEGIES_DIR)):
        if filename in skip_files or not filename.endswith('.py'):
            continue

        filepath = os.path.join(_STRATEGIES_DIR, filename)
        strategy_id = filename[:-3]  # remove .py

        # Try to extract class name and docstring from the file
        name = strategy_id.replace('_', ' ').title()
        description = ''
        strategy_type = 'Custom'

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Extract class name
            class_match = re.search(r'class\s+(\w+)\s*\(', content)
            if class_match:
                class_name = class_match.group(1)
                # Convert CamelCase to readable name
                readable = re.sub(r'(?<!^)(?=[A-Z])', ' ', class_name)
                if readable.lower() not in ('base strategy', 'my strategy'):
                    name = readable

            # Extract docstring (first triple-quoted string after class)
            doc_match = re.search(r'class\s+\w+.*?:\s*\n\s*"""(.*?)"""', content, re.DOTALL)
            if not doc_match:
                doc_match = re.search(r"class\s+\w+.*?:\s*\n\s*'''(.*?)'''", content, re.DOTALL)
            if doc_match:
                description = doc_match.group(1).strip().split('\n')[0]

            if not description:
                description = f'Custom strategy from {filename}'

        except Exception:
            description = f'Custom strategy from {filename}'

        user_strategies.append({
            'id': strategy_id,
            'name': name,
            'description': description,
            'parameters': 'Custom',
            'type': strategy_type,
            'is_user': True,
            'filename': filename,
        })

    return user_strategies


def _get_available_strategies():
    """Return list of all available strategy definitions (built-in + user-created)."""
    strategies = [
        {
            'id': 'sma_crossover',
            'name': 'SMA Crossover',
            'description': 'Moving Average Crossover strategy using 20-period and 50-period Simple Moving Averages. Generates BUY when short MA crosses above long MA, and SELL when it crosses below.',
            'parameters': 'Short Window: 20, Long Window: 50',
            'type': 'Trend Following',
            'is_user': False,
        },
    ]

    # Add user-created strategies
    strategies.extend(_scan_user_strategies())

    return strategies


def _load_strategy_instance(strategy_id):
    """Dynamically load and instantiate a strategy by its ID."""
    if strategy_id == 'sma_crossover':
        return SampleStrategy()

    # Try to load user strategy
    filepath = os.path.join(_STRATEGIES_DIR, f'{strategy_id}.py')
    if not os.path.exists(filepath):
        raise ValueError(f'Strategy file not found: {strategy_id}.py')

    # Dynamic import
    spec = importlib.util.spec_from_file_location(f'openengine.strategies.{strategy_id}', filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the strategy class (must extend BaseStrategy)
    strategy_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type)
                and issubclass(attr, BaseStrategy)
                and attr is not BaseStrategy):
            strategy_class = attr
            break

    if strategy_class is None:
        raise ValueError(f'No BaseStrategy subclass found in {strategy_id}.py')

    return strategy_class()


@main_bp.route('/')
def dashboard():
    """Dashboard / landing page."""
    total_backtests = len(_backtest_results)
    last_run = None
    last_return = None

    if _backtest_results:
        latest = max(_backtest_results.values(), key=lambda x: x['timestamp'])
        last_run = latest['timestamp'].strftime('%Y-%m-%d %H:%M')
        last_return = latest.get('total_return', 0)

    return render_template('dashboard.html',
                           total_backtests=total_backtests,
                           last_run=last_run,
                           last_return=last_return,
                           strategies_count=len(_get_available_strategies()))


@main_bp.route('/backtest')
def backtest_form():
    """Backtest configuration form."""
    strategies = _get_available_strategies()
    return render_template('backtest.html', strategies=strategies)


@main_bp.route('/backtest/run', methods=['POST'])
def run_backtest():
    """Execute a backtest and return results."""
    global _backtest_counter

    try:
        # Parse form data
        symbol = request.form.get('symbol', 'RELIANCE.NS').strip()
        start_date = request.form.get('start_date', '2023-01-01')
        end_date = request.form.get('end_date', '2024-01-01')
        initial_capital = float(request.form.get('initial_capital', 100000))
        strategy_id = request.form.get('strategy', 'sma_crossover')

        # New config fields (with safe defaults)
        brokerage_pct = float(request.form.get('brokerage_pct', 0.0))
        slippage_pct = float(request.form.get('slippage_pct', 0.0))
        sizing_mode_str = request.form.get('sizing_mode', 'fixed_quantity')
        fixed_quantity = int(request.form.get('fixed_quantity', 0))
        percent_of_capital = float(request.form.get('percent_of_capital', 100.0))
        risk_per_trade = float(request.form.get('risk_per_trade', 2.0))
        sl_pct = float(request.form.get('sl_pct', 0.0))
        tp_pct = float(request.form.get('tp_pct', 0.0))

        # Build config
        sizing_map = {
            'fixed_quantity': SizingMode.FIXED_QUANTITY,
            'percent_of_capital': SizingMode.PERCENT_OF_CAPITAL,
            'risk_based': SizingMode.RISK_BASED,
        }
        config = BacktestConfig(
            initial_capital=initial_capital,
            brokerage_pct=brokerage_pct,
            slippage_pct=slippage_pct,
            sizing_mode=sizing_map.get(sizing_mode_str, SizingMode.FIXED_QUANTITY),
            fixed_quantity=fixed_quantity,
            percent_of_capital=percent_of_capital,
            risk_per_trade_pct=risk_per_trade,
            default_sl_pct=sl_pct,
            default_tp_pct=tp_pct,
        )

        # Fetch data
        connector = YahooFinanceConnector()
        data = connector.fetch_data(symbol, start_date, end_date, interval='1d')

        if data is None or data.empty:
            flash('No data returned for the given symbol and date range.', 'error')
            return redirect(url_for('main.backtest_form'))

        # Initialize strategy (dynamically loads user strategies)
        strategy = _load_strategy_instance(strategy_id)

        # Run backtest — returns structured BacktestResult
        backtester = Backtester(data, strategy, config)
        result = backtester.run()

        if result.equity_curve.empty:
            flash('Backtest produced no results. Try a longer date range.', 'error')
            return redirect(url_for('main.backtest_form'))

        # Extract structured data from result
        result_dict = result.to_dict()

        # --- Monte Carlo ---
        mc_enabled = request.form.get('mc_enabled') == '1'
        mc_simulations = int(request.form.get('mc_simulations', 1000))
        mc_data = None
        if mc_enabled and result.trades:
            returns = get_trade_returns(result.trades)
            mc_result = run_monte_carlo(returns, initial_capital, simulations=mc_simulations)
            mc_data = mc_result.to_dict()

        # Find strategy name for display
        strat_name = strategy_id
        for s in _get_available_strategies():
            if s['id'] == strategy_id:
                strat_name = s['name']
                break

        # Store result
        _backtest_counter += 1
        result_id = str(_backtest_counter)
        _backtest_results[result_id] = {
            'id': result_id,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_value': result.metrics.get('final_value', initial_capital),
            'total_return': result.metrics.get('total_return_pct', 0),
            'max_drawdown': result.metrics.get('max_drawdown_pct', 0),
            'total_trades': result.metrics.get('total_trades', 0),
            'equity_dates': result_dict['equity_curve']['dates'],
            'equity_values': result_dict['equity_curve']['values'],
            'drawdown_dates': result_dict['drawdown_series']['dates'],
            'drawdown_values': result_dict['drawdown_series']['values'],
            'trade_log': result_dict['trades'],
            'metrics': result.metrics,
            'mc_data': mc_data,
            'strategy': strat_name,
            'timestamp': datetime.now(),
            # Config snapshot for display
            'config': {
                'brokerage_pct': brokerage_pct,
                'slippage_pct': slippage_pct,
                'sizing_mode': sizing_mode_str,
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
            },
        }

        return redirect(url_for('main.results', result_id=result_id))

    except Exception as e:
        traceback.print_exc()
        flash(f'Backtest failed: {str(e)}', 'error')
        return redirect(url_for('main.backtest_form'))


@main_bp.route('/backtest/oos', methods=['POST'])
def run_oos_backtest():
    """Execute an out-of-sample backtest and return train/test comparison results."""
    global _backtest_counter

    try:
        # Parse form data (same as regular backtest)
        symbol = request.form.get('symbol', 'RELIANCE.NS').strip()
        start_date = request.form.get('start_date', '2023-01-01')
        end_date = request.form.get('end_date', '2024-01-01')
        initial_capital = float(request.form.get('initial_capital', 100000))
        strategy_id = request.form.get('strategy', 'sma_crossover')

        brokerage_pct = float(request.form.get('brokerage_pct', 0.0))
        slippage_pct = float(request.form.get('slippage_pct', 0.0))
        sizing_mode_str = request.form.get('sizing_mode', 'fixed_quantity')
        fixed_quantity = int(request.form.get('fixed_quantity', 0))
        percent_of_capital = float(request.form.get('percent_of_capital', 100.0))
        risk_per_trade = float(request.form.get('risk_per_trade', 2.0))
        sl_pct = float(request.form.get('sl_pct', 0.0))
        tp_pct = float(request.form.get('tp_pct', 0.0))

        # OOS-specific fields
        split_method = request.form.get('split_method', 'percentage')
        train_pct = float(request.form.get('train_pct', 70.0))
        split_date_val = request.form.get('split_date', '')
        optimize = request.form.get('oos_optimize') == '1'

        # Build configs
        sizing_map = {
            'fixed_quantity': SizingMode.FIXED_QUANTITY,
            'percent_of_capital': SizingMode.PERCENT_OF_CAPITAL,
            'risk_based': SizingMode.RISK_BASED,
        }
        config = BacktestConfig(
            initial_capital=initial_capital,
            brokerage_pct=brokerage_pct,
            slippage_pct=slippage_pct,
            sizing_mode=sizing_map.get(sizing_mode_str, SizingMode.FIXED_QUANTITY),
            fixed_quantity=fixed_quantity,
            percent_of_capital=percent_of_capital,
            risk_per_trade_pct=risk_per_trade,
            default_sl_pct=sl_pct,
            default_tp_pct=tp_pct,
        )

        oos_config = OOSConfig(
            enabled=True,
            split_method=split_method,
            split_date=split_date_val,
            train_pct=train_pct,
            optimize=optimize,
        )

        # Fetch data
        connector = YahooFinanceConnector()
        data = connector.fetch_data(symbol, start_date, end_date, interval='1d')

        if data is None or data.empty:
            flash('No data returned for the given symbol and date range.', 'error')
            return redirect(url_for('main.backtest_form'))

        # Load strategy
        strategy = _load_strategy_instance(strategy_id)

        # Run OOS backtest
        oos_result = run_out_of_sample_backtest(data, strategy, config, oos_config)

        # Serialize
        oos_dict = oos_result.to_dict()

        # --- Monte Carlo on Test Phase ---
        mc_enabled = request.form.get('mc_enabled') == '1'
        mc_simulations = int(request.form.get('mc_simulations', 1000))
        mc_data = None
        if mc_enabled and oos_result.test_result.trades:
            returns = get_trade_returns(oos_result.test_result.trades)
            mc_result = run_monte_carlo(returns, initial_capital, simulations=mc_simulations)
            mc_data = mc_result.to_dict()

        # Find strategy name
        strat_name = strategy_id
        for s in _get_available_strategies():
            if s['id'] == strategy_id:
                strat_name = s['name']
                break

        # Store result
        _backtest_counter += 1
        result_id = str(_backtest_counter)
        _backtest_results[result_id] = {
            'id': result_id,
            'is_oos': True,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'strategy': strat_name,
            'timestamp': datetime.now(),
            'oos_data': oos_dict,
            'train_metrics': oos_dict['train_metrics'],
            'test_metrics': oos_dict['test_metrics'],
            'best_params': oos_dict['best_params'],
            'warnings': oos_dict['warnings'],
            'split_info': oos_dict['split_info'],
            'train_equity': oos_dict['train_equity_curve'],
            'test_equity': oos_dict['test_equity_curve'],
            'train_drawdown': oos_dict['train_drawdown'],
            'test_drawdown': oos_dict['test_drawdown'],
            'train_trade_log': oos_dict['train_trade_log'],
            'test_trade_log': oos_dict['test_trade_log'],
            'mc_data': mc_data,
            # For dashboard display
            'total_return': oos_dict['test_metrics'].get('total_return_pct', 0),
            'final_value': oos_dict['test_metrics'].get('final_value', initial_capital),
            'max_drawdown': oos_dict['test_metrics'].get('max_drawdown_pct', 0),
            'total_trades': oos_dict['test_metrics'].get('total_trades', 0),
        }

        return redirect(url_for('main.oos_results', result_id=result_id))

    except Exception as e:
        traceback.print_exc()
        flash(f'OOS Backtest failed: {str(e)}', 'error')
        return redirect(url_for('main.backtest_form'))


@main_bp.route('/oos-results/<result_id>')
def oos_results(result_id):
    """Display out-of-sample backtest results."""
    result = _backtest_results.get(result_id)
    if not result or not result.get('is_oos'):
        flash('OOS Result not found.', 'error')
        return redirect(url_for('main.dashboard'))

    return render_template('oos_results.html',
                           result=result,
                           train_metrics=result['train_metrics'],
                           test_metrics=result['test_metrics'],
                           best_params=result['best_params'],
                           warnings=result['warnings'],
                           split_info=result['split_info'],
                           train_trade_log=result['train_trade_log'],
                           test_trade_log=result['test_trade_log'],
                           train_eq_dates_json=json.dumps(result['train_equity']['dates']),
                           train_eq_values_json=json.dumps(result['train_equity']['values']),
                           test_eq_dates_json=json.dumps(result['test_equity']['dates']),
                           test_eq_values_json=json.dumps(result['test_equity']['values']),
                           train_dd_dates_json=json.dumps(result['train_drawdown']['dates']),
                           train_dd_values_json=json.dumps(result['train_drawdown']['values']),
                           test_dd_dates_json=json.dumps(result['test_drawdown']['dates']),
                           test_dd_values_json=json.dumps(result['test_drawdown']['values']),
                           mc_data=result.get('mc_data'))


@main_bp.route('/results/<result_id>')
def results(result_id):
    """Display backtest results."""
    result = _backtest_results.get(result_id)
    if not result:
        flash('Result not found.', 'error')
        return redirect(url_for('main.dashboard'))

    return render_template('results.html',
                           result=result,
                           equity_dates_json=json.dumps(result['equity_dates']),
                           equity_values_json=json.dumps(result['equity_values']),
                           drawdown_dates_json=json.dumps(result.get('drawdown_dates', [])),
                           drawdown_values_json=json.dumps(result.get('drawdown_values', [])),
                           metrics=result.get('metrics', {}),
                           trade_log_json=json.dumps(result.get('trade_log', [])),
                           mc_data=result.get('mc_data'))


@main_bp.route('/strategies')
def strategies():
    """List available strategies."""
    strategy_list = _get_available_strategies()
    return render_template('strategies.html', strategies=strategy_list)


@main_bp.route('/strategies/add')
def add_strategy():
    """Show the strategy code editor."""
    return render_template('add_strategy.html', default_code=_DEFAULT_STRATEGY_CODE)


@main_bp.route('/strategies/save', methods=['POST'])
def save_strategy():
    """Save a user-created strategy to the strategies folder."""
    try:
        strategy_name = request.form.get('strategy_name', '').strip()
        strategy_desc = request.form.get('strategy_desc', '').strip()
        code = request.form.get('code', '')

        if not strategy_name:
            flash('Strategy name is required.', 'error')
            return redirect(url_for('main.add_strategy'))

        if not code.strip():
            flash('Strategy code cannot be empty.', 'error')
            return redirect(url_for('main.add_strategy'))

        # Convert name to valid Python filename
        filename = re.sub(r'[^a-zA-Z0-9_]', '_', strategy_name.lower().strip())
        filename = re.sub(r'_+', '_', filename).strip('_')

        if not filename:
            flash('Invalid strategy name.', 'error')
            return redirect(url_for('main.add_strategy'))

        filepath = os.path.join(_STRATEGIES_DIR, f'{filename}.py')

        # Validate the code
        validation = _validate_code(code)
        if not validation['valid']:
            flash(f'Code validation failed: {validation["message"]}', 'error')
            return redirect(url_for('main.add_strategy'))

        # Write the file
        with open(filepath, 'w') as f:
            f.write(code)

        flash(f'Strategy "{strategy_name}" saved as {filename}.py', 'success')
        return redirect(url_for('main.strategies'))

    except Exception as e:
        traceback.print_exc()
        flash(f'Failed to save strategy: {str(e)}', 'error')
        return redirect(url_for('main.add_strategy'))


@main_bp.route('/strategies/validate', methods=['POST'])
def validate_strategy():
    """Validate strategy code without saving."""
    data = request.get_json()
    code = data.get('code', '')
    result = _validate_code(code)
    return jsonify(result)


@main_bp.route('/strategies/delete/<strategy_id>', methods=['POST'])
def delete_strategy(strategy_id):
    """Delete a user-created strategy file."""
    # Safety: don't allow deleting built-in strategies
    protected = {'sample_strategy', 'base_strategy', '__init__'}
    if strategy_id in protected:
        flash('Cannot delete built-in strategies.', 'error')
        return redirect(url_for('main.strategies'))

    filepath = os.path.join(_STRATEGIES_DIR, f'{strategy_id}.py')
    if os.path.exists(filepath):
        os.remove(filepath)
        flash(f'Strategy {strategy_id} deleted.', 'success')
    else:
        flash('Strategy file not found.', 'error')

    return redirect(url_for('main.strategies'))


def _validate_code(code):
    """Validate Python code for strategy correctness."""
    if not code.strip():
        return {'valid': False, 'message': 'Code cannot be empty.'}

    # Check syntax
    try:
        compile(code, '<strategy>', 'exec')
    except SyntaxError as e:
        return {'valid': False, 'message': f'Syntax error on line {e.lineno}: {e.msg}'}

    # Check for required class
    if 'BaseStrategy' not in code:
        return {'valid': False, 'message': 'Code must contain a class that extends BaseStrategy.'}

    if 'generate_signals' not in code:
        return {'valid': False, 'message': 'Strategy class must implement generate_signals() method.'}

    if 'class ' not in code:
        return {'valid': False, 'message': 'Code must define a class.'}

    return {'valid': True, 'message': 'Code is valid! Strategy class found with generate_signals method.'}


@main_bp.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'openengine-web'})
