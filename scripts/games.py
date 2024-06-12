from agents import Nomad
import nashpy as nash
import numpy as np

        
def fighting_game(agent1: Nomad, agent2: Nomad, alpha):
    if agent1.spice >= agent2.spice:
            weak_agent = agent2
            strong_agent = agent1
    else:
        weak_agent = agent1
        strong_agent = agent2

    if agent1.tribe != agent2.tribe: 
        strong_agent_payoffs = np.array([[0, weak_agent.spice - alpha*strong_agent.spice], [weak_agent.spice, weak_agent.spice - (alpha/2)*strong_agent.spice]])
        weak_agent_payoffs = np.array([[0, -weak_agent.spice], [-weak_agent.spice, -weak_agent.spice]])

        fight = nash.Game(strong_agent_payoffs, weak_agent_payoffs)

        equilibria = list(fight.support_enumeration())
        print(equilibria)

        strong_agent_strategy = np.argmax(equilibria[0])
        weak_agent_strategy = np.argmax(equilibria[1])

        strong_agent.spice += strong_agent_payoffs[strong_agent_strategy, weak_agent_strategy]
        weak_agent.spice += weak_agent_payoffs[weak_agent_strategy, strong_agent_strategy]
    
    else:
        payoff = (strong_agent.spice - weak_agent.spice)//2
        weak_agent.spice += payoff
        strong_agent.spice -= payoff

    print(f"After the game, the stronger agent has {strong_agent.spice} spice.")
    print(f"After the game, the weaker agent has {weak_agent.spice} spice.")

def pure_eq(game: np.ndarray):
    best_response = np.argmax(game, axis=1), np.argmax(game, axis=2)
    return game[0, best_response[0], best_response[1]], game[1, best_response[0], best_response[1]]