"""
Simple visualization to understand MCTS decision-making.
Creates a heatmap showing which board positions MCTS explores most.
"""

import numpy as np
import matplotlib.pyplot as plt
from agents import MCTSAgent, MCTSNode
from go_search_problem import GoProblem
import time

def create_visualization():
    """
    Create a simple heatmap showing MCTS action preferences
    """
    print("Running MCTS analysis...")
    
    # Initialize
    problem = GoProblem()
    initial_state = problem.start_state
    board_size = 5
    
    agent = MCTSAgent(c=np.sqrt(2))
    time_limit = 2.0
    
    # Run MCTS search
    start_time = time.time()
    cushion = 0.2
    
    root = MCTSNode(state=initial_state, parent=None, action=None)
    
    while time.time() - start_time < time_limit - cushion:
        leaf = agent._select(root)
        children = agent._expand(leaf)
        for child in children:
            result = agent._simulate(child)
            agent._backpropagate(child, result)
    
    # Create heatmap data
    visit_map = np.zeros((board_size, board_size))
    winrate_map = np.zeros((board_size, board_size))
    
    for child in root.children:
        action = child.action
        if action < board_size * board_size:  # Not a pass move
            row = action // board_size
            col = action % board_size
            visit_map[row, col] = child.visits
            if child.visits > 0:
                winrate_map[row, col] = child.value / child.visits
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap 1: Visit counts
    im1 = ax1.imshow(visit_map, cmap='YlOrRd', interpolation='nearest')
    ax1.set_title('MCTS Visit Count per Position\n(2 seconds of search)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column', fontsize=12)
    ax1.set_ylabel('Row', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='Number of Visits')
    
    # Add visit counts as text
    for i in range(board_size):
        for j in range(board_size):
            if visit_map[i, j] > 0:
                ax1.text(j, i, f'{int(visit_map[i, j])}', 
                        ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    
    # Heatmap 2: Win rates
    im2 = ax2.imshow(winrate_map, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title('MCTS Win Rate per Position\n(from BLACK perspective)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column', fontsize=12)
    ax2.set_ylabel('Row', fontsize=12)
    plt.colorbar(im2, ax=ax2, label='Win Rate')
    
    # Add win rates as text
    for i in range(board_size):
        for j in range(board_size):
            if visit_map[i, j] > 0:
                ax2.text(j, i, f'{winrate_map[i, j]:.2f}', 
                        ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mcts_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'mcts_analysis.png'")
    
    # Print summary
    total_sims = int(visit_map.sum())
    print(f"\nTotal simulations: {total_sims}")
    
    print("\nTop 5 most-explored positions:")
    top_actions = []
    for i in range(board_size):
        for j in range(board_size):
            if visit_map[i, j] > 0:
                top_actions.append((visit_map[i, j], winrate_map[i, j], (i, j)))
    top_actions.sort(reverse=True)
    
    for visits, winrate, (i, j) in top_actions[:5]:
        print(f"  Position ({i},{j}): {int(visits)} visits ({visits/total_sims*100:.1f}%), {winrate:.1%} win rate")

if __name__ == "__main__":
    create_visualization()
