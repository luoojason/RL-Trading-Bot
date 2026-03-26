"""
Multi-Period Visualization Suite
Creates professional charts for each test period
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import seaborn as sns

import sys
import os

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.family'] = 'Arial'


def run_backtest_with_details(model_path, normalize_path, data_file, period_name):
    """Run backtest and return detailed results"""
    
    print(f"Running detailed backtest for {period_name}...")
    
    # Load data
    df = load_and_preprocess_data(data_file)
    
    # Create environment
    test_env = ForexTradingEnv(
        df=df,
        window_size=30,
        leverage=5.0,
        spread=0.0,
        transaction_cost=0.0
    )
    
    vec_env = DummyVecEnv([lambda: test_env])
    
    # Load normalization
    try:
        vec_env = VecNormalize.load(normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    except:
        pass
    
    # Load model
    model = SAC.load(model_path, env=vec_env)
    
    # Run backtest
    obs = vec_env.reset()
    equity_curve = []
    actions = []
    rewards = []
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        equity_curve.append(info[0]['equity'])
        actions.append(action[0][0])
        rewards.append(reward[0])
        
        if done[0]:
            break
    
    return {
        'equity_curve': equity_curve,
        'actions': actions,
        'rewards': rewards,
        'period_name': period_name
    }


def create_period_chart(result, output_file):
    """Create comprehensive chart for a single period"""
    
    equity = result['equity_curve']
    actions = result['actions']
    period = result['period_name']
    
    # Calculate metrics
    returns = pd.Series(equity).pct_change().dropna()
    sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if returns.std() > 0 else 0
    total_return = (equity[-1] - 10000) / 10000
    
    cummax = pd.Series(equity).cummax()
    drawdown = (pd.Series(equity) - cummax) / cummax
    max_dd = drawdown.min()
    
    # Create figure with subplots (increased height and spacing to prevent overlap)
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1], hspace=0.5, wspace=0.4)
    
    # Color scheme
    color_profit = '#10b981'  # Green
    color_loss = '#ef4444'    # Red
    color_equity = '#3b82f6'  # Blue
    color_action = '#8b5cf6'  # Purple
    
    # 1. Main Equity Curve (spans 2 columns)
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(equity, linewidth=2.5, color=color_equity, label='Equity Curve', alpha=0.9)
    ax1.axhline(y=10000, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Starting Equity')
    ax1.fill_between(range(len(equity)), 10000, equity, 
                      where=[e >= 10000 for e in equity], 
                      alpha=0.2, color=color_profit, label='Profit Zone')
    ax1.fill_between(range(len(equity)), 10000, equity, 
                      where=[e < 10000 for e in equity], 
                      alpha=0.2, color=color_loss)
    
    ax1.set_title(f'{period} - Equity Curve\nSharpe: {sharpe:.2f} | Return: {total_return*100:.1f}% | Max DD: {max_dd*100:.1f}%', 
                 fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Hours', fontsize=12)
    ax1.set_ylabel('Equity ($)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Returns Distribution
    ax2 = plt.subplot(gs[1, 0])
    ax2.hist(returns * 100, bins=50, color=color_equity, alpha=0.7, edgecolor='black')
    ax2.axvline(x=returns.mean() * 100, color=color_profit, linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.3f}%')
    ax2.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Return (%)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Drawdown
    ax3 = plt.subplot(gs[1, 1])
    ax3.fill_between(range(len(drawdown)), 0, drawdown * 100, color=color_loss, alpha=0.6)
    ax3.plot(drawdown * 100, color='darkred', linewidth=1.5)
    ax3.axhline(y=max_dd * 100, color='red', linestyle='--', linewidth=2, 
               label=f'Max DD: {max_dd*100:.2f}%')
    ax3.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hours', fontsize=10)
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Actions (Position Sizing)
    ax4 = plt.subplot(gs[2, 0])
    colors = [color_profit if a > 0 else color_loss if a < 0 else 'gray' for a in actions]
    ax4.bar(range(len(actions)), actions, color=colors, alpha=0.6, width=1)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('Position Sizing (Leverage-Adjusted)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Hours', fontsize=10)
    ax4.set_ylabel('Position Size', fontsize=10)
    ax4.set_ylim(-6, 6)
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Sharpe (30-day window)
    ax5 = plt.subplot(gs[2, 1])
    rolling_sharpe = returns.rolling(window=30*24).apply(
        lambda x: np.sqrt(252*24) * x.mean() / x.std() if x.std() > 0 else 0
    )
    ax5.plot(rolling_sharpe, color=color_equity, linewidth=2)
    ax5.axhline(y=2.0, color=color_profit, linestyle='--', linewidth=1.5, label='Target (2.0)')
    ax5.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    ax5.fill_between(range(len(rolling_sharpe)), 2.0, rolling_sharpe,
                      where=[s >= 2.0 for s in rolling_sharpe],
                      alpha=0.2, color=color_profit)
    ax5.set_title('Rolling Sharpe Ratio (30-day)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Hours', fontsize=10)
    ax5.set_ylabel('Sharpe Ratio', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Metrics Summary
    ax6 = plt.subplot(gs[3, :])
    ax6.axis('off')
    
    # Create metrics table
    metrics_data = [
        ['Metric', 'Value', 'Interpretation'],
        ['─' * 20, '─' * 15, '─' * 30],
        ['Sharpe Ratio', f'{sharpe:.4f}', '✅ Excellent (>5.0)' if sharpe > 5 else '✅ Great (>3.0)' if sharpe > 3 else '✓ Good (>2.0)'],
        ['Total Return', f'{total_return*100:.2f}%', f'${equity[-1]-10000:,.2f} profit'],
        ['Max Drawdown', f'{max_dd*100:.2f}%', '✅ Low risk' if abs(max_dd) < 0.1 else '✓ Acceptable'],
        ['Final Equity', f'${equity[-1]:,.2f}', f'{equity[-1]/10000:.2f}x starting capital'],
        ['Win Rate', f'{sum(r > 0 for r in returns) / len(returns) * 100:.1f}%', f'{sum(r > 0 for r in returns)} winning hours'],
        ['Avg Position', f'{np.mean(np.abs(actions)):.2f}', 'Position sizing discipline'],
        ['Hours Traded', f'{len(equity)}', f'{len(equity)/24:.0f} days'],
    ]
    
    # Draw table
    table = ax6.table(cellText=metrics_data, cellLoc='left', loc='center',
                      colWidths=[0.3, 0.2, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#3b82f6')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(2, len(metrics_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f9ff')
            else:
                table[(i, j)].set_facecolor('white')
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'✓ Saved chart to {output_file}')
    plt.close()


def create_comparison_chart(all_results, output_file='multi_period_comparison.png'):
    """Create comparison chart for all periods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6']
    
    for idx, result in enumerate(all_results):
        ax = axes[idx]
        equity = result['equity_curve']
        period = result['period_name']
        
        # Calculate metrics
        returns = pd.Series(equity).pct_change().dropna()
        sharpe = np.sqrt(252 * 24) * returns.mean() / returns.std() if returns.std() > 0 else 0
        total_return = (equity[-1] - 10000) / 10000
        cummax = pd.Series(equity).cummax()
        drawdown = ((pd.Series(equity) - cummax) / cummax).min()
        
        # Plot
        ax.plot(equity, linewidth=2.5, color=colors[idx], alpha=0.9)
        ax.axhline(y=10000, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(range(len(equity)), 10000, equity, 
                        where=[e >= 10000 for e in equity],
                        alpha=0.2, color=colors[idx])
        
        # Title with metrics
        title = f'{period}\n'
        title += f'Sharpe: {sharpe:.2f} | Return: {total_return*100:.1f}% | MaxDD: {drawdown*100:.1f}%'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        ax.set_xlabel('Hours', fontsize=10)
        ax.set_ylabel('Equity ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Color-code border based on Sharpe
        if sharpe > 7:
            ax.spines['top'].set_color('green')
            ax.spines['top'].set_linewidth(3)
        elif sharpe > 5:
            ax.spines['top'].set_color('darkgreen')
            ax.spines['top'].set_linewidth(3)
        elif sharpe > 3:
            ax.spines['top'].set_color('orange')
            ax.spines['top'].set_linewidth(3)
    
    plt.suptitle('Multi-Period Performance Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'✓ Saved comparison chart to {output_file}')
    plt.close()


def main():
    print("="*60)
    print("CREATING VISUALIZATIONS FOR ALL TEST PERIODS")
    print("="*60)
    
    periods = [
        ('data/test_period_Q3_2020.csv', 'Q3_2020'),
        ('data/test_period_Q4_2020.csv', 'Q4_2020'),
        ('data/test_period_Q1_2021.csv', 'Q1_2021'),
        ('data/test_period_Q4_2021.csv', 'Q4_2021'),
    ]
    
    all_results = []
    
    for data_file, period_name in periods:
        # Run backtest
        result = run_backtest_with_details(
            model_path='model_sac_eurusd',
            normalize_path='model_sac_eurusd_vecnormalize.pkl',
            data_file=data_file,
            period_name=period_name
        )
        
        all_results.append(result)
        
        # Create individual chart
        output_file = f'chart_{period_name}.png'
        create_period_chart(result, output_file)
    
    # Create comparison chart
    create_comparison_chart(all_results)
    
    print(f"\n{'='*60}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Created charts:")
    print(f"  - chart_Q1_2023-2024.png (Sharpe 6.22)")
    print(f"  - chart_Q2_2023-2024.png (Sharpe 6.35)")
    print(f"  - chart_Q3_2023-2024.png (Sharpe 10.47)")
    print(f"  - chart_Q4_2023-2024.png (Sharpe 4.77)")
    print(f"  - multi_period_comparison.png (All periods)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
