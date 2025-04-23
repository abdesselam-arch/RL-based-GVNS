import itertools
import random
import math
import copy

# from matplotlib import pyplot as plt

from PersonalModules.utilities import dijkstra

def adaptive_crossover(parent1, parent2, generation, max_generations):
    """
    Adaptive crossover that becomes more exploitative as generations progress.
    Early generations: More randomness to explore solution space
    Later generations: Focus on using the better parts of each parent
    """
    # Calculate progress through generations (0 to 1)
    progress = min(1.0, generation / (max_generations * 0.7))
    
    # Choose the solution with fewer nodes as the primary parent
    if sum(len(route) for route in parent1) < sum(len(route) for route in parent2):
        better_parent = parent1
        other_parent = parent2
    else:
        better_parent = parent2
        other_parent = parent1
    
    child = []
    for gene1, gene2 in zip(better_parent, other_parent):
        # Adaptive probability - more bias toward better parent as generations progress
        # Early generations: ~50% chance to take from either parent
        # Later generations: Up to 80% chance to take from better parent
        take_from_better = random.random() < (0.5 + 0.3 * progress)
        
        # Choose route from appropriate parent
        if take_from_better:
            child.append(gene1)
        else:
            child.append(gene2)
    
    return child

def path_based_crossover(parent1, parent2, sink, custom_range):
    """
    Smarter crossover that focuses on path quality rather than random selection.
    Selects routes based on their effectiveness at connecting to the sink.
    """
    child = []
    
    # Evaluate each route's quality in both parents
    for i in range(len(parent1)):
        route1 = parent1[i]
        route2 = parent2[i]
        
        # Calculate metrics for each route
        route1_length = len(route1)
        route2_length = len(route2)
        
        # Check if routes connect to sink
        route1_connects = route_connects_to_sink(route1, sink, custom_range)
        route2_connects = route_connects_to_sink(route2, sink, custom_range)
        
        # Prioritize connectivity, then shortest route
        if route1_connects and not route2_connects:
            child.append(route1)
        elif route2_connects and not route1_connects:
            child.append(route2)
        elif route1_connects and route2_connects:
            # Both connect, choose the shorter one
            if route1_length <= route2_length:
                child.append(route1)
            else:
                child.append(route2)
        else:
            # Neither connects, select randomly
            if random.random() < 0.5:
                child.append(route1)
            else:
                child.append(route2)
    
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

def aggressive_pruning(solution, sink, custom_range):
    """Aggressively remove redundant relay nodes while maintaining connectivity."""
    pruned_solution = copy.deepcopy(solution)
    
    # First pass: remove obviously redundant relays
    for route_idx, route in enumerate(pruned_solution):
        if len(route) <= 1:
            continue
        
        # Try removing relays one by one
        i = 1
        while i < len(route):
            temp_route = route.copy()
            removed_node = temp_route.pop(i)
            
            still_connected = is_connected(temp_route, custom_range)
            still_reaches_sink = route_connects_to_sink(temp_route, sink, custom_range)
            
            if still_connected and still_reaches_sink:
                route.pop(i)
                # Don't increment i since we've removed an element
            else:
                i += 1
    
    # Second pass: try skipping intermediate nodes
    for route_idx, route in enumerate(pruned_solution):
        if len(route) <= 2:  # Need at least 3 nodes to skip intermediates
            continue
        
        i = 1
        while i < len(route) - 1:  # Check pairs of nodes excluding the last one
            # Try connecting node i-1 directly to node i+1
            if math.dist(route[i-1], route[i+1]) < custom_range:
                # Can skip the intermediate node
                temp_route = route.copy()
                temp_route.pop(i)
                
                # Ensure this doesn't break sink connectivity
                if route_connects_to_sink(temp_route, sink, custom_range):
                    route.pop(i)
                    # Don't increment i
                else:
                    i += 1
            else:
                i += 1
    
    # Third pass: try route merging for common paths
    # This is a more complex optimization that could be implemented to
    # identify and merge common partial paths between routes
    
    return pruned_solution

def intelligent_mutation(solution, free_slots, custom_range, sink, mutation_rate=0.3):
    """
    Enhanced mutation with multiple intelligent strategies:
    1. More aggressive pruning
    2. Targeted bridge building 
    3. Route optimization
    4. Path straightening
    """
    mutated_solution = copy.deepcopy(solution)
    
    # Only proceed if there are free slots available
    if not free_slots:
        return mutated_solution
    
    # Vary mutation approach based on random chance
    mutation_type = random.random()
    
    # 1. PRUNING STRATEGY (40% chance)
    if mutation_type < 0.4:
        # Choose a random route to optimize
        route_index = random.randint(0, len(mutated_solution) - 1)
        route = mutated_solution[route_index]
        
        if len(route) > 1:
            # Try to systematically remove nodes
            for i in range(len(route)-1, 0, -1):  # Start from the end, excluding sentinel
                # Try removing this node
                temp_route = route.copy()
                removed_node = temp_route.pop(i)
                
                # Check if connectivity is maintained
                if is_connected(temp_route, custom_range) and route_connects_to_sink(temp_route, sink, custom_range):
                    # Safe to remove this node
                    mutated_solution[route_index].pop(i)
                    free_slots.append(removed_node)
                    return mutated_solution
    
    # 2. BRIDGE BUILDING STRATEGY (30% chance)
    elif mutation_type < 0.7:
        # Find routes that don't connect to sink
        disconnected_routes = []
        
        for i, route in enumerate(mutated_solution):
            if not route_connects_to_sink(route, sink, custom_range):
                disconnected_routes.append(i)
        
        if disconnected_routes:
            # Choose a disconnected route to fix
            route_index = random.choice(disconnected_routes)
            current_end = mutated_solution[route_index][-1]
            
            # Try to build an efficient bridge to the sink
            sink_connectors = [node for node in free_slots if math.dist(node, sink) < custom_range]
            
            if sink_connectors:
                # Find best connector (closest to current end)
                if any(math.dist(current_end, node) < custom_range for node in sink_connectors):
                    # Direct connection possible
                    best_connector = min(sink_connectors, 
                                        key=lambda node: math.dist(current_end, node))
                    
                    mutated_solution[route_index].append(best_connector)
                    free_slots.remove(best_connector)
                    return mutated_solution
                else:
                    # Need a two-hop bridge
                    # Find first hop candidates
                    first_hop_candidates = [node for node in free_slots 
                                          if math.dist(current_end, node) < custom_range]
                    
                    if first_hop_candidates:
                        # For each first hop, find potential second hops
                        bridges = []
                        
                        for first_hop in first_hop_candidates:
                            for connector in sink_connectors:
                                if math.dist(first_hop, connector) < custom_range:
                                    # Found a viable two-hop bridge
                                    distance = math.dist(current_end, first_hop) + math.dist(first_hop, connector)
                                    bridges.append((first_hop, connector, distance))
                        
                        if bridges:
                            # Choose the shortest bridge
                            best_bridge = min(bridges, key=lambda x: x[2])
                            first_hop, second_hop = best_bridge[0], best_bridge[1]
                            
                            # Add the bridge
                            mutated_solution[route_index].append(first_hop)
                            mutated_solution[route_index].append(second_hop)
                            
                            free_slots.remove(first_hop)
                            free_slots.remove(second_hop)
                            return mutated_solution
    
    # 3. PATH STRAIGHTENING STRATEGY (20% chance)
    elif mutation_type < 0.9:
        route_index = random.randint(0, len(mutated_solution) - 1)
        route = mutated_solution[route_index]
        
        if len(route) >= 3:  # Need at least 3 nodes to straighten
            # Try to replace a zigzag with a more direct path
            for i in range(len(route) - 2):
                # Check if we can find a better node to connect i and i+2
                start_node = route[i]
                end_node = route[i+2]
                
                # Only attempt if current path is not direct
                if math.dist(start_node, end_node) > custom_range:
                    # Find candidates that can connect start and end
                    candidates = [node for node in free_slots 
                                if math.dist(start_node, node) < custom_range 
                                and math.dist(node, end_node) < custom_range]
                    
                    if candidates:
                        # Choose the best candidate (closest to straight line)
                        def distance_from_line(point, start, end):
                            # Calculate distance from point to line segment
                            line_length = math.dist(start, end)
                            if line_length == 0:
                                return math.dist(point, start)
                            
                            t = max(0, min(1, ((point[0] - start[0]) * (end[0] - start[0]) + 
                                               (point[1] - start[1]) * (end[1] - start[1])) / (line_length ** 2)))
                            
                            projection = (start[0] + t * (end[0] - start[0]), 
                                         start[1] + t * (end[1] - start[1]))
                            
                            return math.dist(point, projection)
                        
                        best_candidate = min(candidates, 
                                           key=lambda node: distance_from_line(node, start_node, end_node))
                        
                        # Replace the middle node
                        old_middle = route[i+1]
                        route[i+1] = best_candidate
                        
                        # Update free slots
                        free_slots.remove(best_candidate)
                        free_slots.append(old_middle)
                        return mutated_solution
    
    # 4. RANDOM MODIFICATION (10% chance)
    else:
        # Standard random add/remove approach
        route_index = random.randint(0, len(mutated_solution) - 1)
        
        if random.random() < 0.5 and free_slots:  # Add a relay
            chosen_slot = random.choice(free_slots)
            mutated_solution[route_index].append(chosen_slot)
            free_slots.remove(chosen_slot)
        elif len(mutated_solution[route_index]) > 1:  # Remove a relay if possible
            relay_index = random.randint(1, len(mutated_solution[route_index]) - 1)
            relay = mutated_solution[route_index][relay_index]
            
            temp_route = mutated_solution[route_index].copy()
            temp_route.pop(relay_index)
            
            if is_connected(temp_route, custom_range) and route_connects_to_sink(temp_route, sink, custom_range):
                mutated_solution[route_index].pop(relay_index)
                free_slots.append(relay)
    
    return mutated_solution

def weighted_evaluate(solution, sink, grid, free_slots, mesh_size, custom_range, generation, max_generations):
    """
    Adaptive fitness function that adjusts weights based on generation progress.
    Early generations: Focus on establishing connectivity
    Later generations: Focus on minimizing relays
    """
    # Extract sentinels and relays
    sinked_sentinels = [route[0] for route in solution]
    sinked_relays = [relay for route in solution for relay in route[1:]]
    
    # Count total relay nodes
    relay_count = len(sinked_relays)
    
    # Calculate progress ratio (0 to 1)
    progress = min(1.0, generation / (max_generations * 0.7))
    
    # Adjust weights based on progress
    relay_weight = 0.3 + (0.4 * progress)  # Increases from 0.3 to 0.7
    diameter_weight = 0.3 - (0.1 * progress)  # Decreases from 0.3 to 0.2
    
    # Check connectivity
    connectivity_penalty = 0
    
    for route in solution:
        # Check if route connects to sink
        if not route_connects_to_sink(route, sink, custom_range):
            connectivity_penalty += 1000
        
        # Check internal connectivity
        if not is_connected(route, custom_range):
            connectivity_penalty += 500
    
    # Calculate network metrics using dijkstra
    distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
    
    # Network disconnected penalty
    network_disconnected_penalty = 2000 if 999 in sentinel_bman else 0
    
    # Calculate fitness
    fitness = (relay_weight * relay_count) + (diameter_weight * (cal_bman / mesh_size)) + connectivity_penalty + network_disconnected_penalty
    
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
        # Try different strategies in priority order:
        
        # 1. Find direct connections to sink (best case)
        direct_connectors = [node for node in available_slots 
                           if math.dist(node, sink) < custom_range 
                           and math.dist(current_node, node) < custom_range]
        
        if direct_connectors:
            # Choose the one closest to current_node to minimize hop distance
            best_connector = min(direct_connectors, key=lambda node: math.dist(current_node, node))
            route.append(best_connector)
            available_slots.remove(best_connector)
            return route
        
        # 2. Try to make the longest possible jump toward the sink
        # This reduces the number of hops needed
        candidates = [node for node in available_slots 
                     if math.dist(current_node, node) < custom_range]
        
        if not candidates:
            # No valid next hop found, route cannot be completed
            break
        
        # Choose the candidate that gets us closest to the sink
        # But also consider the progress made (distance traveled)
        best_candidates = []
        for candidate in candidates:
            # Calculate how much closer we get to the sink
            current_to_sink = math.dist(current_node, sink)
            candidate_to_sink = math.dist(candidate, sink)
            progress_toward_sink = current_to_sink - candidate_to_sink
            
            # Only consider candidates that make substantial progress
            if progress_toward_sink > 0:
                best_candidates.append((candidate, progress_toward_sink))
        
        # If we found candidates that make progress
        if best_candidates:
            # Sort by progress and take the best one
            best_candidates.sort(key=lambda x: x[1], reverse=True)
            best_candidate = best_candidates[0][0]
        else:
            # Fall back to just closest to sink
            best_candidate = min(candidates, key=lambda node: math.dist(node, sink))
        
        route.append(best_candidate)
        available_slots.remove(best_candidate)
        current_node = best_candidate
        hops += 1
        
        # Check if we can now reach the sink
        if math.dist(current_node, sink) < custom_range:
            break
    
    return route

def diverse_initial_population(population_size, sinkless_sentinels, free_slots, max_hops_number, custom_range, grid, sink):
    """
    Create a diverse initial population using multiple strategies.
    """
    population = []
    
    # Create a copy of free_slots
    original_free_slots = free_slots.copy()
    
    # Strategy distribution
    num_minimal = int(population_size * 0.4)  # 40% minimal routes
    num_balanced = int(population_size * 0.3)  # 30% balanced routes
    num_diverse = population_size - num_minimal - num_balanced  # 30% diverse routes
    
    # 1. MINIMAL ROUTES STRATEGY - focus on fewest relays
    for _ in range(num_minimal):
        available_slots = original_free_slots.copy()
        sentinel_solution = []
        
        for sentinel in sinkless_sentinels:
            route = create_efficient_route(sentinel, sink, available_slots, custom_range, max_hops_number)
            sentinel_solution.append(route)
        
        population.append(sentinel_solution)
    
    # 2. BALANCED ROUTES STRATEGY - consider both hops and total distance
    for _ in range(num_balanced):
        available_slots = original_free_slots.copy()
        sentinel_solution = []
        
        for sentinel in sinkless_sentinels:
            route = [sentinel]
            current_node = sentinel
            hops = 0
            
            # Check if sentinel can directly connect to sink
            if math.dist(sentinel, sink) < custom_range:
                continue
            
            while current_node != sink and hops < max_hops_number:
                # Find all candidates within range
                candidates = [node for node in available_slots 
                             if math.dist(current_node, node) < custom_range]
                
                if not candidates:
                    break
                
                # Balance between:
                # 1. Getting closer to sink
                # 2. Making substantial movement
                best_candidate = None
                best_score = float('-inf')
                
                for candidate in candidates:
                    # How much closer to sink
                    progress = math.dist(current_node, sink) - math.dist(candidate, sink)
                    # Distance moved from current
                    movement = math.dist(current_node, candidate)
                    
                    # Balance score - reward progress and longer jumps
                    score = progress + (0.3 * movement)
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                
                if best_candidate:
                    route.append(best_candidate)
                    available_slots.remove(best_candidate)
                    current_node = best_candidate
                    hops += 1
                    
                    if math.dist(current_node, sink) < custom_range:
                        break
                else:
                    break
            
            sentinel_solution.append(route)
        
        population.append(sentinel_solution)
    
    # 3. DIVERSE ROUTES STRATEGY - intentionally different from others
    for _ in range(num_diverse):
        available_slots = original_free_slots.copy()
        sentinel_solution = []
        
        for sentinel in sinkless_sentinels:
            route = [sentinel]
            current_node = sentinel
            hops = 0
            
            if math.dist(sentinel, sink) < custom_range:
                continue
            
            # Add some randomness in path selection
            while current_node != sink and hops < max_hops_number:
                candidates = [node for node in available_slots 
                             if math.dist(current_node, node) < custom_range]
                
                if not candidates:
                    break
                
                # Mix randomness with direction
                if random.random() < 0.3:  # 30% chance of random selection
                    best_candidate = random.choice(candidates)
                else:
                    # Choose based on progress to sink
                    best_candidate = min(candidates, key=lambda node: math.dist(node, sink))
                
                route.append(best_candidate)
                available_slots.remove(best_candidate)
                current_node = best_candidate
                hops += 1
                
                if math.dist(current_node, sink) < custom_range:
                    break
            
            sentinel_solution.append(route)
        
        # Randomly add a few extra relays for diversity
        extra_relays = random.randint(1, 3)
        unused_slots = [slot for slot in original_free_slots if slot not in sum(sentinel_solution, [])]
        
        if unused_slots and extra_relays > 0:
            for _ in range(min(extra_relays, len(unused_slots))):
                route_idx = random.randint(0, len(sentinel_solution) - 1)
                slot = random.choice(unused_slots)
                # Insert at random position (not first)
                if len(sentinel_solution[route_idx]) > 1:
                    insert_pos = random.randint(1, len(sentinel_solution[route_idx]))
                    sentinel_solution[route_idx].insert(insert_pos, slot)
                else:
                    sentinel_solution[route_idx].append(slot)
                unused_slots.remove(slot)
        
        population.append(sentinel_solution)
    
    # Prune all solutions
    for i in range(len(population)):
        population[i] = aggressive_pruning(population[i], sink, custom_range)
    
    return population

def genetic_algorithm(population_size, generations, sink, sinkless_sentinels, free_slots, max_hops_number, custom_range, mesh_size):
    """Enhanced genetic algorithm with adaptive operators and improved strategies."""
    
    # Create a copy of free_slots to avoid modifying the original
    free_slots_original = free_slots.copy()
    
    grid = len(free_slots) + len(sinkless_sentinels) + 1
    print("The grid =", grid)
    
    # Track best solution and its fitness
    best_solution = None
    best_fitness = float('inf')
    best_generation = 0
    
    # Track fitness statistics
    fitness_per_generation = []
    
    # Create diverse initial population
    population = diverse_initial_population(population_size, sinkless_sentinels, free_slots_original, max_hops_number, custom_range, grid, sink)
    
    # Track stagnation
    generations_without_improvement = 0
    
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
            
            # Evaluate fitness - use adaptive weights based on generation
            fitness = weighted_evaluate(solution, sink, grid, current_free_slots, mesh_size, custom_range, generation, generations)
            fitness_scores.append(fitness)
            free_slots_tracking.append(current_free_slots)
        
        # Find the best solution in this generation
        current_best_index = fitness_scores.index(min(fitness_scores))
        current_best_fitness = fitness_scores[current_best_index]
        
        # Update overall best solution if better than previous best
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = copy.deepcopy(population[current_best_index])
            best_generation = generation
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1
        
        # Record best fitness for this generation
        fitness_per_generation.append(current_best_fitness)
        
        # Print progress
        print(f"  Best fitness: {current_best_fitness:.2f}")
        print(f"  Relay count in best solution: {sum(len(route)-1 for route in population[current_best_index])}")
        
        # Check for early convergence (optional)
        if generations_without_improvement > generations // 3:
            print(f"Early convergence detected after {generation+1} generations")
            break
        
        # Apply aggressive pruning to the best solution
        population[current_best_index] = aggressive_pruning(population[current_best_index], sink, custom_range)
        
        # Selection - tournament selection with size adaptation
        # Larger tournaments in later generations = more selective pressure
        tournament_size = max(3, min(5, 3 + int(generation / (generations / 3))))
        
        parent_indices = []
        for _ in range(2):  # Select 2 parents
            tournament = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament]
            winner_idx = tournament[tournament_fitness.index(min(tournament_fitness))]
            parent_indices.append(winner_idx)
        
        parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
        
        # Ensure we have the correct free slots for each parent
        free_slots_parent1 = free_slots_tracking[parent_indices[0]]
        free_slots_parent2 = free_slots_tracking[parent_indices[1]]
        
        # Use specialized crossover based on generation
        if generation < generations * 0.3:  # First 30% - use adaptive crossover
            child1 = adaptive_crossover(parent1, parent2, generation, generations)
            child2 = adaptive_crossover(parent2, parent1, generation, generations)
        else:  # Later generations - use path-based crossover
            child1 = path_based_crossover(parent1, parent2, sink, custom_range)
            child2 = path_based_crossover(parent2, parent1, sink, custom_range)
        
        # Enhanced mutation
        child1 = intelligent_mutation(child1, free_slots_parent1.copy(), custom_range, sink)
        child2 = intelligent_mutation(child2, free_slots_parent2.copy(), custom_range, sink)
        
        # Aggressive pruning on both children
        child1 = aggressive_pruning(child1, sink, custom_range)
        child2 = aggressive_pruning(child2, sink, custom_range)
        
        # Population replacement strategy:
        # 1. Always replace worst solution
        # 2. Second replacement depends on diversity needs
        
        worst_index = fitness_scores.index(max(fitness_scores))
        population[worst_index] = child1
        
        # For second child, replace either second worst or a randomly selected solution
        if generations_without_improvement > generations // 5:
            # If stagnating, replace a random solution to maintain diversity
            replace_idx = random.randint(0, len(population) - 1)
            while replace_idx == worst_index:  # Don't replace the one we just updated
                replace_idx = random.randint(0, len(population) - 1)
        else:
            # Otherwise replace second worst
            sorted_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)
            replace_idx = sorted_indices[1]  # Second worst
        
        population[replace_idx] = child2
    
    print(f"\nBest solution found in generation {best_generation+1} with fitness {best_fitness:.2f}")
    
    # If no good solution was found, use the best from final population
    if best_solution is None:
        final_fitness_scores = []
        for solution in population:
            current_free_slots = free_slots_original.copy()
            used_relays = [relay for route in solution for relay in route[1:]]
            for relay in used_relays:
                if relay in current_free_slots:
                    current_free_slots.remove(relay)
            fitness = weighted_evaluate(solution, sink, grid, current_free_slots, mesh_size, custom_range, generations-1, generations)
            final_fitness_scores.append(fitness)
        
        best_solution_index = final_fitness_scores.index(min(final_fitness_scores))
        best_solution = population[best_solution_index]
    
    # Do a final aggressive pruning of the best solution
    best_solution = aggressive_pruning(best_solution, sink, custom_range)
    
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