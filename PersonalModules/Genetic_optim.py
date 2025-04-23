import itertools
import random
import math
import copy

# from matplotlib import pyplot as plt

from PersonalModules.utilities import dijkstra

def crossover(parent1, parent2):
    # Perform crossover between two parents using Uniform Crossover
    
    # Choose the solution with fewer nodes as the primary parent
    if sum(len(route) for route in parent1) < sum(len(route) for route in parent2):
        primary_parent = parent1
        secondary_parent = parent2
    else:
        primary_parent = parent2
        secondary_parent = parent1
    
    child = []
    for gene1, gene2 in zip(primary_parent, secondary_parent):
        # Randomly select the gene from either parent with a 50% probability
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)
    
    return child

def is_connected(route, custom_range):
    """Check if all nodes in a route are connected within the custom range."""
    for i in range(len(route) - 1):
        if math.dist(route[i], route[i+1]) > custom_range:
            return False
    return True

def route_connects_to_sink(route, sink, custom_range):
    """Check if any node in the route can connect to the sink."""
    for node in route:
        if node == sink or math.dist(node, sink) < custom_range:
            return True
    return False

def prune_redundant_relays(solution, sink, custom_range):
    """Remove redundant relay nodes while maintaining connectivity."""
    pruned_solution = copy.deepcopy(solution)
    
    for route_idx, route in enumerate(pruned_solution):
        # Skip if route has only the sentinel
        if len(route) <= 1:
            continue
        
        # Start from the second node (first relay) and check if each can be removed
        i = 1
        while i < len(route):
            # Temporarily remove the relay
            temp_route = route.copy()
            removed_node = temp_route.pop(i)
            
            # Check if route remains connected without this relay
            still_connected = is_connected(temp_route, custom_range)
            
            # Check if route still connects to sink
            still_reaches_sink = route_connects_to_sink(temp_route, sink, custom_range)
            
            # If removing the relay maintains connectivity, keep it removed
            if still_connected and still_reaches_sink:
                route.pop(i)
                # Don't increment i since we've removed an element
            else:
                # This relay is necessary, move to the next one
                i += 1
    
    return pruned_solution

def mutate(solution, free_slots, custom_range, sink):
    """Perform mutation with a focus on minimizing relays while maintaining connectivity."""
    mutated_solution = copy.deepcopy(solution)
    
    # Only proceed if there are free slots available
    if not free_slots:
        return mutated_solution
    
    # First, try to remove unnecessary relay nodes (50% chance)
    if random.random() < 0.5:
        route_index = random.randint(0, len(mutated_solution) - 1)
        route = mutated_solution[route_index]
        
        if len(route) > 1:  # Only try to remove if there's more than just the sentinel
            # Choose a random relay node (not the sentinel)
            relay_indices = list(range(1, len(route)))
            if relay_indices:
                relay_index = random.choice(relay_indices)
                relay = route[relay_index]
                
                # Temporarily remove the relay
                temp_route = route.copy()
                temp_route.pop(relay_index)
                
                # Check if connectivity is maintained
                if is_connected(temp_route, custom_range) and route_connects_to_sink(temp_route, sink, custom_range):
                    # Safe to remove this relay
                    mutated_solution[route_index].pop(relay_index)
                    free_slots.append(relay)
                    return mutated_solution
    
    # If we didn't remove a node or removing wasn't successful, try regular mutation
    # Randomly select a route to modify
    sentinel_index = random.randint(0, len(mutated_solution) - 1)
    
    # Check if this route connects to the sink
    route_connects = route_connects_to_sink(mutated_solution[sentinel_index], sink, custom_range)
    
    if not route_connects:
        # Route doesn't connect to sink, try to add a connecting node
        # Find the current end of the route
        if not mutated_solution[sentinel_index]:
            current_end = mutated_solution[sentinel_index][0]  # Sentinel
        else:
            current_end = mutated_solution[sentinel_index][-1]
        
        # Find nodes that can connect to the sink
        sink_connectors = [node for node in free_slots if math.dist(node, sink) < custom_range]
        
        if sink_connectors:
            # Find nodes that can connect to both current end and sink
            bridge_nodes = []
            for connector in sink_connectors:
                if math.dist(current_end, connector) < custom_range:
                    bridge_nodes.append(connector)
            
            if bridge_nodes:
                # Add a bridge node to connect to sink
                chosen_node = min(bridge_nodes, key=lambda node: math.dist(node, sink))
                mutated_solution[sentinel_index].append(chosen_node)
                free_slots.remove(chosen_node)
                return mutated_solution
            
            # If no direct bridge, try two-hop connection (only rarely)
            if random.random() < 0.2:
                for fs in free_slots:
                    if math.dist(current_end, fs) < custom_range:
                        for connector in sink_connectors:
                            if math.dist(fs, connector) < custom_range:
                                # Add two-hop bridge
                                mutated_solution[sentinel_index].append(fs)
                                mutated_solution[sentinel_index].append(connector)
                                free_slots.remove(fs)
                                free_slots.remove(connector)
                                return mutated_solution
    
    # If we're still here, try regular mutation - add/remove a random node
    if free_slots and random.random() < 0.7:  # Bias toward adding nodes for connectivity
        # Add a new relay
        chosen_slot = random.choice(free_slots)
        mutated_solution[sentinel_index].append(chosen_slot)
        free_slots.remove(chosen_slot)
    elif len(mutated_solution[sentinel_index]) > 1:  # Only remove if there's more than just the sentinel
        # Remove a relay (not the sentinel)
        relay_index = random.randint(1, len(mutated_solution[sentinel_index]) - 1)
        relay = mutated_solution[sentinel_index][relay_index]
        
        # Only remove if it doesn't break connectivity
        temp_route = mutated_solution[sentinel_index].copy()
        temp_route.pop(relay_index)
        
        if is_connected(temp_route, custom_range) and route_connects_to_sink(temp_route, sink, custom_range):
            mutated_solution[sentinel_index].pop(relay_index)
            free_slots.append(relay)
    
    return mutated_solution

def evaluate(solution, sink, grid, free_slots, mesh_size, custom_range):
    """Evaluate solution fitness with a strong emphasis on minimizing relays while ensuring connectivity."""
    # Extract sentinels and relays
    sinked_sentinels = [route[0] for route in solution]
    sinked_relays = [relay for route in solution for relay in route[1:]]
    
    # Count total relay nodes (this is what we want to minimize)
    relay_count = len(sinked_relays)
    
    # Check connectivity - all routes must connect to sink
    connectivity_penalty = 0
    disconnected_routes = 0
    
    for route in solution:
        # Check if route connects to sink
        if not route_connects_to_sink(route, sink, custom_range):
            connectivity_penalty += 1000
            disconnected_routes += 1
        
        # Check internal connectivity
        if not is_connected(route, custom_range):
            connectivity_penalty += 500
    
    # Use dijkstra to calculate network metrics
    distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
    
    # If any sentinel is disconnected (999 in sentinel_bman), add huge penalty
    network_disconnected_penalty = 2000 if 999 in sentinel_bman else 0
    
    # Calculate fitness - heavily weight relay count and connectivity
    # The formula balances:
    # 1. Number of relays (lower is better)
    # 2. Network diameter (lower is better)
    # 3. Connectivity (must be maintained)
    fitness = (0.5 * relay_count) + (0.2 * (cal_bman / mesh_size)) + connectivity_penalty + network_disconnected_penalty
    
    return fitness

def create_efficient_route(sentinel, sink, available_slots, custom_range, max_hops):
    """Create an efficient route from sentinel to sink with minimal relays."""
    route = [sentinel]
    current_node = sentinel
    hops = 0
    
    # Check if sentinel can directly connect to sink
    if math.dist(sentinel, sink) < custom_range:
        return route
    
    while current_node != sink and hops < max_hops:
        # Find nodes that can directly connect to sink
        direct_connectors = [node for node in available_slots 
                            if math.dist(node, sink) < custom_range 
                            and math.dist(current_node, node) < custom_range]
        
        if direct_connectors:
            # Choose the one closest to current_node to minimize hop distance
            best_connector = min(direct_connectors, key=lambda node: math.dist(current_node, node))
            route.append(best_connector)
            available_slots.remove(best_connector)
            return route
        
        # No direct connector found, find the best next hop
        candidates = [node for node in available_slots 
                     if math.dist(current_node, node) < custom_range]
        
        if not candidates:
            # No valid next hop found, route cannot be completed
            break
        
        # Choose the candidate that gets us closest to the sink
        best_candidate = min(candidates, key=lambda node: math.dist(node, sink))
        route.append(best_candidate)
        available_slots.remove(best_candidate)
        current_node = best_candidate
        hops += 1
        
        # Check if we can now reach the sink
        if math.dist(current_node, sink) < custom_range:
            break
    
    return route

def initial_population(population_size, sinkless_sentinels, free_slots, max_hops_number, custom_range, grid, sink):
    """Create initial population with focus on minimal relays with connectivity."""
    population = []
    
    # Create a copy of free_slots to avoid modifying the original
    original_free_slots = free_slots.copy()
    
    for _ in range(population_size):
        # Reset available slots for each new solution
        available_slots = original_free_slots.copy()
        sentinel_solution = []
        
        # First, create minimal efficient routes for each sentinel
        for sentinel in sinkless_sentinels:
            # Create an efficient route with minimal relays
            route = create_efficient_route(sentinel, sink, available_slots, custom_range, max_hops_number)
            sentinel_solution.append(route)
        
        # Add some diversity by adding some random relays (only in 30% of solutions)
        if random.random() < 0.3:
            # Calculate number of relays to add (small number)
            relays_to_add = random.randint(1, max(1, int(len(original_free_slots) * 0.05)))
            
            # Get list of unused slots
            used_slots = [node for route in sentinel_solution for node in route[1:]]
            unused_slots = [slot for slot in original_free_slots if slot not in used_slots]
            
            if unused_slots and relays_to_add > 0:
                for _ in range(min(relays_to_add, len(unused_slots))):
                    # Add to a random route
                    route_idx = random.randint(0, len(sentinel_solution) - 1)
                    slot = random.choice(unused_slots)
                    sentinel_solution[route_idx].append(slot)
                    unused_slots.remove(slot)
        
        # Add solution to population
        population.append(sentinel_solution)
    
    # Prune any redundant relays in all solutions
    for i in range(len(population)):
        population[i] = prune_redundant_relays(population[i], sink, custom_range)
    
    return population

def genetic_algorithm(population_size, generations, sink, sinkless_sentinels, free_slots, max_hops_number, custom_range, mesh_size):
    """Genetic algorithm focusing on minimizing relay nodes while maintaining connectivity."""
    
    # Create a copy of free_slots to avoid modifying the original
    free_slots_original = free_slots.copy()
    
    grid = len(free_slots) + len(sinkless_sentinels) + 1
    print("The grid =", grid)
    
    # Track best solution and its fitness
    best_solution = None
    best_fitness = float('inf')
    
    # Track fitness statistics
    fitness_per_generation = []
    
    # Create initial population with efficient routes
    population = initial_population(population_size, sinkless_sentinels, free_slots_original, max_hops_number, custom_range, grid, sink)
    
    for generation in range(generations):
        print(f'Generation {generation+1}')
        
        # Evaluate all solutions
        fitness_scores = []
        free_slots_tracking = []
        
        for solution in population:
            # Create a working copy of free slots for evaluation
            current_free_slots = free_slots_original.copy()
            used_relays = [relay for route in solution for relay in route[1:]]
            
            # Remove used relays from free slots
            for relay in used_relays:
                if relay in current_free_slots:
                    current_free_slots.remove(relay)
            
            # Evaluate fitness
            fitness = evaluate(solution, sink, grid, current_free_slots, mesh_size, custom_range)
            fitness_scores.append(fitness)
            free_slots_tracking.append(current_free_slots)
        
        # Find the best solution in this generation
        current_best_index = fitness_scores.index(min(fitness_scores))
        current_best_fitness = fitness_scores[current_best_index]
        
        # Update overall best solution if better than previous best
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = copy.deepcopy(population[current_best_index])
        
        # Record best fitness for this generation
        fitness_per_generation.append(current_best_fitness)
        
        # Apply pruning to the best solution to remove any redundant relays
        population[current_best_index] = prune_redundant_relays(population[current_best_index], sink, custom_range)
        
        # Selection - tournament selection for parents
        parent_indices = []
        tournament_size = 3
        
        for _ in range(2):  # Select 2 parents
            tournament = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner_idx = tournament[tournament_fitness.index(min(tournament_fitness))]
            parent_indices.append(winner_idx)
        
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
        
        # Ensure we have the correct free slots for each parent
        free_slots_parent1 = free_slots_tracking[parent_indices[0]]
        free_slots_parent2 = free_slots_tracking[parent_indices[1]]
        
        # Create children through crossover
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)
        
        # Mutation with focus on minimizing relays
        child1 = mutate(child1, free_slots_parent1.copy(), custom_range, sink)
        child2 = mutate(child2, free_slots_parent2.copy(), custom_range, sink)
        
        # Prune any redundant relays in both children
        child1 = prune_redundant_relays(child1, sink, custom_range)
        child2 = prune_redundant_relays(child2, sink, custom_range)
        
        # Replace worst solutions
        worst_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:2]
        population[worst_indices[0]] = child1
        population[worst_indices[1]] = child2
    
    # If no good solution was found, use the best from final population
    if best_solution is None:
        final_fitness_scores = []
        for solution in population:
            current_free_slots = free_slots_original.copy()
            used_relays = [relay for route in solution for relay in route[1:]]
            for relay in used_relays:
                if relay in current_free_slots:
                    current_free_slots.remove(relay)
            fitness = evaluate(solution, sink, grid, current_free_slots, mesh_size, custom_range)
            final_fitness_scores.append(fitness)
        
        best_solution_index = final_fitness_scores.index(min(final_fitness_scores))
        best_solution = population[best_solution_index]
    
    # Do a final pruning of the best solution
    best_solution = prune_redundant_relays(best_solution, sink, custom_range)
    
    # Extract results from best solution
    sinked_sentinels = [route[0] for route in best_solution]
    sinked_relays = [relay for route in best_solution for relay in route[1:]]
    
    # Calculate remaining free slots
    all_used_nodes = sinked_sentinels + sinked_relays
    free_slots_remaining = [slot for slot in free_slots_original if slot not in all_used_nodes and slot != sink]
    
    # Calculate min hop counts
    min_hop_counts = calculate_min_hop_count(sink, sinked_relays, mesh_size)
    sinked_relays = list(zip(sinked_relays, min_hop_counts))
    
    print(f"\nOptimized solution found with {len(sinked_relays)} relay nodes")
    print(f"Connected sentinels: {len(sinked_sentinels)} out of {len(sinkless_sentinels)}")
    
    return sinked_sentinels, sinked_relays, free_slots_remaining, True, False

def calculate_min_hop_count(sink, sinked_relays, mesh_size):
    min_hop_counts = []
    for relay in sinked_relays:
        # Calculate Manhattan distance (hop count) from relay to the sink
        distance = abs(sink[0] - relay[0]) + abs(sink[1] - relay[1])
        distance = distance / mesh_size
        min_hop_counts.append(distance)
    return min_hop_counts