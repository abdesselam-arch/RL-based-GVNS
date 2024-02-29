import itertools
import random
import math

from matplotlib import pyplot as plt

from PersonalModules.utilities import bellman_ford, get_stat

def plot_best_solution(generations_range, fitness_per_generation):
    # Plot the evolution of best fitness scores over generations
    plt.plot(generations_range, fitness_per_generation)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Evolution of Best Fitness Scores')
    plt.show()

def plot_fitness_scores(all_fitness_scores):
     # Plot the fitness scores of all solutions in every generation
    for gen, fitness_scores in enumerate(all_fitness_scores):
        plt.scatter([gen] * len(fitness_scores), fitness_scores, c='b', alpha=0.5)

def has_non_border_neighbor(node, border_nodes, sentinel_solution):
    for route in sentinel_solution:
        for i, route_node in enumerate(route):
            if route_node == node:
                neighbor_indices = [i - 1, i + 1]
                for index in neighbor_indices:
                    if 0 <= index < len(route):
                        neighbor = route[index]
                        if neighbor not in border_nodes:
                            return True
    return False

def deploy_new_node(border_node, free_slots, custom_range, sentinel_solution, border_nodes):
    # Calculate the distances from the border node to each free slot
    distances = {slot: math.dist(border_node, slot) for slot in free_slots}
    
    # Sort free slots based on distance
    sorted_free_slots = sorted(free_slots, key=lambda slot: distances[slot])

    for route in sentinel_solution:
        for i, route_node in enumerate(route):
            if route_node == border_node:
                neighbor_indices = [i - 1, i + 1]
                for index in neighbor_indices:
                    if 0 <= index < len(route):
                        neighbor = route[index]
                        if neighbor not in border_nodes:
                            # Deploy a new node close to the current border node
                            nearby_candidates = [slot for slot in sorted_free_slots if math.dist(border_node, slot) < custom_range]
                            if nearby_candidates:
                                new_node = nearby_candidates[0]
                                sentinel_solution.append(new_node)  # Insert the new node after the current border node
                                free_slots.remove(new_node)  # Remove the new node from free_slots
                                print(f"Deployed a new node {new_node} close to border node {border_node}")
                                return

def initial_population(population_size, sinkless_sentinels, free_slots, max_hops_number, custom_range, sink):
    population = []
    
    for _ in range(population_size):
        sentinel_solution = []
        
        for sentinel in sinkless_sentinels:
            route = [sentinel]
            current_node = sentinel

            # Calculate initial distance from the sink
            initial_distance = math.dist(current_node, sink)
            
            # Track the number of relays connected to the sink
            relays_connected_to_sink = 0

            while current_node != sink and relays_connected_to_sink < 4 and relays_connected_to_sink > 1 :
                nearby_candidates = [node for node in free_slots if math.dist(current_node, node) < custom_range]

                # Filter candidates to prioritize nodes farther from the sink
                filtered_candidates = [node for node in nearby_candidates if math.dist(node, sink) > initial_distance]
                
                if not filtered_candidates:
                    # If no candidates are available, break the loop
                    break
                
                # Choose the nearest candidate from the filtered list
                nearest_candidate = min(filtered_candidates, key=lambda node: math.dist(node, sink))
                
                # Update current node and remove chosen candidate from free slots
                current_node = nearest_candidate
                free_slots.remove(nearest_candidate)
                
                route.append(nearest_candidate)

                # If the nearest candidate is a relay, increment the count
                if nearest_candidate != sink:
                    relays_connected_to_sink += 1

            # Add the route to the solution for the current sentinel
            sentinel_solution.append(route)

        # Check for sets of four nodes with distances less than or equal to custom_range
        '''for i in range(len(sentinel_solution) - 3):
            for j in range(i + 1, len(sentinel_solution) - 2):
                for k in range(j + 1, len(sentinel_solution) - 1):
                    for l in range(k + 1, len(sentinel_solution)):
                        nodes = [sentinel_solution[i][-1], sentinel_solution[j][-1], sentinel_solution[k][-1], sentinel_solution[l][-1]]
                        distances = [math.dist(node1, node2) for node1 in nodes for node2 in nodes if node1 != node2]
                        if all(distance <= custom_range for distance in distances):
                            # Randomly choose one node to delete from the solution
                            node_to_delete = random.choice(nodes)
                            sentinel_solution.remove(node_to_delete)'''

        # Ensure that for every node in the border, there are three nodes close to it
        border_nodes = []
        for route in sentinel_solution:
            for i, node in enumerate(route):
                if node != sink:
                    neighbors = [route[j] for j in range(max(0,i-1), min(len(route), i+2)) if j!=i]
                    if len(neighbors) < 3:
                        border_nodes.append(node)

        # Keep track of the routes from each border node to the sink
        routes_from_border_to_sink = {node: [] for node in border_nodes}
        for route in sentinel_solution:
            for i, node in enumerate(route):
                if node in border_nodes:
                    routes_from_border_to_sink[node].append(route)

        # Delete excess routes from each border node to the sink
        for node, routes in routes_from_border_to_sink.items():
            if len(routes) > 2:
                # Randomly choose routes to keep
                routes_to_keep = random.sample(routes, 2)
                for route in routes:
                    if route not in routes_to_keep:
                        # Add the nodes from the route back to free slots
                        for node_to_free in route:
                            if node_to_free != sink:
                                free_slots.append(node_to_free)
                        # Remove the route from the solution
                        sentinel_solution.remove(route)

        for border_node in border_nodes:
            if has_non_border_neighbor(border_node, border_nodes, sentinel_solution):
                print(f"Border node {border_node} has a non-border neighbor.")
            else:
                deploy_new_node(border_node, free_slots, custom_range, sentinel_solution, border_nodes)

        # Ensure that corners of free slots have nodes deployed in them
        corners = [(min(free_slots, key=lambda x: x[0])[0], min(free_slots, key=lambda x: x[1])[1]),
                   (min(free_slots, key=lambda x: x[0])[0], max(free_slots, key=lambda x: x[1])[1]),
                   (max(free_slots, key=lambda x: x[0])[0], min(free_slots, key=lambda x: x[1])[1]),
                   (max(free_slots, key=lambda x: x[0])[0], max(free_slots, key=lambda x: x[1])[1])]
        for corner in corners:
            if corner in free_slots:
                for route in sentinel_solution:
                    if corner not in route:
                        route.append(corner)
                        free_slots.remove(corner)
                        break

        # Construct the list of nodes in all routes
        routes = [node for route in sentinel_solution for node in route]
        
        # Delete unnecessary nodes deployed, preserving connectivity
        nodes_to_remove = set()
        for node in free_slots:
            if any(math.dist(node, route_node) < custom_range for route_node in routes):
                nodes_to_remove.add(node)
        free_slots = [node for node in free_slots if node not in nodes_to_remove]

        # Add the solution for current sentinel to the population
        population.append(sentinel_solution)
    
    return population

def crossover(parent1, parent2):
    # Perform crossover between two parents using Uniform Crossover
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        # Randomly select the gene from either parent with a 50% probability
        if random.random() < 0.5:
            child.append(gene1)
        else:
            child.append(gene2)
    print('Success! Crossover operation complete')
    return child

def crossover2(parent1, parent2):
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
    print('Success! Crossover operation complete')
    return child

def mutate(solution, free_slots, custom_range):
    # Perform mutation on the solution
    mutated_solution = solution.copy()
    sentinel_index = random.randint(0, len(mutated_solution) - 1)
    current_node = mutated_solution[sentinel_index][-1]
    
    while len(free_slots) > 0:
        nearby_candidates = [node for node in free_slots if math.dist(current_node, node) < custom_range]
        if not nearby_candidates:
            break
        chosen_node = random.choice(nearby_candidates)
        mutated_solution[sentinel_index].append(chosen_node)
        free_slots.remove(chosen_node)
        current_node = chosen_node

    print('Success! Mutation operation complete')
    return mutated_solution

def mutate2(solution, free_slots, custom_range):
    # Perform mutation on the solution
    
    mutated_solution = solution.copy()
    sentinel_index = random.randint(0, len(mutated_solution) - 1)
    
    # Randomly remove nodes from the current route
    route_to_mutate = mutated_solution[sentinel_index]
    for _ in range(len(route_to_mutate)):
        if random.random() < 0.5 and len(route_to_mutate) > 1:
            # Remove a randomly selected node from the route
            node_to_remove = random.choice(route_to_mutate[1:])  # Exclude the sentinel node
            route_to_mutate.remove(node_to_remove)
            free_slots.append(node_to_remove)
    
    print('Success! Mutation operation complete')
    return mutated_solution

def evaluate(solution, sink, sinked_relays, grid, free_slots, sinked_sentinels):
    # Evaluate the fitness of the solution
    # Two objectives diameter, number of relays
    distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
    
    # Penalize solutions based on the total number of deployed nodes
    total_nodes = sum(len(route) for route in solution)
    fitness = (0.4 * total_nodes / math.sqrt(grid)) + (0.3 * len(sinked_relays)) + (0.3 * cal_bman)
    
    return fitness

def calculate_fitness(solution, sink, grid_size):
    total_distance = 0
    for route in solution:
        # Check if the route contains at least one relay (excluding the sentinel node)
        if len(route) > 1:
            for i in range(len(route) - 1):
                total_distance += math.dist(route[i], route[i+1])
            # Add distance from the last relay to the sink
            total_distance += math.dist(route[-1], sink)
        else:
            # If the route only contains the sentinel node, consider it as having infinite distance
            total_distance += grid_size  # A large value to penalize invalid routes
    # Normalize fitness by dividing by grid size
    normalized_fitness = total_distance / grid_size
    return normalized_fitness

def calculate_min_hop_count(sink, sinked_relays, mesh_size):
    min_hop_counts = []
    for relay in sinked_relays:
        # Calculate Manhattan distance (hop count) from relay to the sink
        distance = abs(sink[0] - relay[0]) + abs(sink[1] - relay[1])
        distance = distance / mesh_size
        min_hop_counts.append(distance)
    return min_hop_counts

def genetic_algorithm(population_size, generations, sink, sinkless_sentinels, free_slots, max_hops_number, custom_range, mesh_size):

    First_time = True
    sentinels = sinkless_sentinels[:]
    grid = len(free_slots) + len(sinkless_sentinels) + 1
    print("The grid =", grid)
    sinked_sentinels, sinked_relays, sinkless_relays, candidate_slots = [], [], [], []
    found_forbidden, ERROR = False, False

    fitness_per_generation, all_fitness_scores = [], []

    population = initial_population(population_size, sinkless_sentinels, free_slots, max_hops_number, custom_range, sink)

    for generation in range(generations):

        print(f'Generation {generation+1}')

        # Evaluate the fitness of each solution in the population
        fitness_scores = [evaluate(solution, sink, sinked_relays, grid, free_slots, sinked_sentinels) for solution in population]
        print(fitness_scores)
        all_fitness_scores.append(fitness_scores)  # Store fitness scores of all solutions

        # Select parents based on their fitness
        parents_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[:2]
        parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]

        # Perform crossover and mutation to generate a new solution
        child = crossover2(parent1, parent2)
        child = mutate(parent1, free_slots, custom_range)

        child_relays = [relay for route in child for relay in route[1:]]
        sinked_relays.extend(child_relays)

        # Append the best fitness score of this generation to fitness_per_generation
        fitness_per_generation.append(max(fitness_scores))

        # Replace the least fit solution in the population with the new child
        min_fitness_index = fitness_scores.index(max(fitness_scores))
        population[min_fitness_index] = child

        all_fitness_scores= []

    # Evaluate the final population and select the best solution
    fitness_scores = [calculate_fitness(solution, sink, grid) for solution in population]
    best_solution_index = fitness_scores.index(max(fitness_scores))
    best_solution = population[best_solution_index]

    # Extract relevant variables to match the outputs of greedy_algorithm
    sinked_sentinels = [route[0] for route in best_solution]
    sinked_relays = [relay for route in best_solution for relay in route[1:]]
    free_slots_remaining = [slot for slot in free_slots if slot not in sum(best_solution, [])]

    # Construct the list of nodes in all routes
    routes = [node for route in best_solution for node in route]

    # Delete unnecessary nodes deployed, preserving connectivity
    nodes_to_remove = set()
    for node in free_slots:
        if any(math.dist(node, route_node) < custom_range for route_node in routes):
            nodes_to_remove.add(node)
    free_slots = [node for node in free_slots if node not in nodes_to_remove]
   
    min_hop_counts = calculate_min_hop_count(sink, sinked_relays, mesh_size)
    sinked_relays = list(zip(sinked_relays, min_hop_counts))

    print(f'sinked relays: {sinked_relays}')

    # Ensure dimensions match by creating a range of generations of the same length as best_fitness_per_generation
    generations_range = range(len(fitness_per_generation))
    #plot_best_solution(generations_range, fitness_per_generation)
    #plot_fitness_scores(all_fitness_scores)

    print('\nSinked Sentinels\n',sinked_sentinels)
    print('\nSinked Relays\n', sinked_relays)

    Finished = True  # Placeholder, update based on your termination criteria
    ERROR = False  # Placeholder, update based on error conditions

    return sinked_sentinels, sinked_relays, free_slots_remaining, Finished, ERROR
