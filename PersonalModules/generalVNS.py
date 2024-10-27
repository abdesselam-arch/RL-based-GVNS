import random
import math
from PersonalModules.utilities import epsilon_constraints, dijkstra, get_Diameter

def GVNS(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, max_iterations, alpha, beta):
    best_solution = sinked_relays.copy()
    print(f"Starting GVNS with max_iterations: {max_iterations}")
    best_fitness = epsilon_constraints(grid, free_slots, sink, sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
    print('----------------------------------')
    # Shaking
    # temp_free_slots, temp_sinked_relays = shaking(sinked_sentinels, best_solution, free_slots.copy(), custom_range, sink, mesh_size, k=5)
                                                        #sinked_sentinels, sinked_relays, free_slots, custom_range, sink, mesh_size
    num_relays = len(best_solution)
    distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
    diameter = get_Diameter(sentinel_bman, cal_bman, mesh_size)
    print(f"  Number of relays: {num_relays}")
    print(f"  Diameter: {diameter}")
    print(f"  Best fitness: {best_fitness}")
    print('----------------------------------')
    
    no_improvement_count = 0
    max_no_improvement = 10
    
    for iteration in range(max_iterations):
        print(f'The iteration number is: {iteration}')

        # Shaking (moved inside the main loop)
        temp_free_slots, temp_sinked_relays = shaking(sinked_sentinels, best_solution.copy(), free_slots.copy(), custom_range, sink, mesh_size, k=5)

        k = 1
        while k <= 5:  # 5 neighborhoods
            print(f'The iteration number is: {iteration}')
            # Local search
            temp_free_slots, temp_sinked_relays = variable_neighborhood_descent(grid, sink, sinked_sentinels, temp_sinked_relays, temp_free_slots, custom_range, mesh_size, alpha, beta)
            
            # Evaluate the new solution
            new_fitness = epsilon_constraints(grid, temp_free_slots, sink, temp_sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
            
            if new_fitness < best_fitness:
                best_solution = temp_sinked_relays.copy()
                best_fitness = new_fitness
                k = 1  # Reset neighborhood
                no_improvement_count = 0  # Reset counter

            else:
                k += 1  # Move to next neighborhood
        
        no_improvement_count += 1
        print(f"Iteration {iteration + 1}: Best fitness = {best_fitness}")

        # Calculate and print the number of relays and diameter
        num_relays = len(best_solution)
        distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
        diameter = get_Diameter(sentinel_bman, cal_bman, mesh_size)
        print(f'The iteration number is: {iteration}')
        print(f"  Number of relays: {num_relays}")
        print(f"  Diameter: {diameter}")
        print(f"  Best fitness: {best_fitness}")
        print('----------------------------------')

        # Early stopping condition
        if no_improvement_count >= max_no_improvement:
            print(f"Stopping early due to no improvement for {max_no_improvement} iterations.")
            break
    
    return best_solution, free_slots

def shaking(sinked_sentinels, sinked_relays, free_slots, custom_range, sink, mesh_size, k):
    for _ in range(k):
        neighborhood = random.randint(1, 5)
        if neighborhood == 1:
            free_slots, sinked_relays, _, _ = add_next_to_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
        elif neighborhood == 2:
            free_slots, sinked_relays, _, _ = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        elif neighborhood == 3:
            free_slots, sinked_relays, _, _ = delete_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        elif neighborhood == 4:
            free_slots, sinked_relays, _, _ = relocate_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
        elif neighborhood == 5:
            free_slots, sinked_relays, _, _ = add_relay_next_to_sentinel(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
    return free_slots, sinked_relays

def variable_neighborhood_descent(grid, sink, sinked_sentinels, sinked_relays, free_slots, custom_range, mesh_size, alpha, beta):
    improvement = True
    while improvement:
        improvement = False
        for neighborhood in range(1, 6):
            temp_free_slots, temp_sinked_relays = apply_neighborhood(neighborhood, sinked_sentinels, sink, sinked_relays.copy(), free_slots.copy(), custom_range, mesh_size)
            
            if is_better_solution(grid, temp_free_slots, sink, temp_sinked_relays, sinked_sentinels, free_slots, sinked_relays, mesh_size, alpha, beta):
                free_slots, sinked_relays = temp_free_slots, temp_sinked_relays
                improvement = True
                break
    
    return free_slots, sinked_relays

def apply_neighborhood(neighborhood, sinked_sentinels, sink, sinked_relays, free_slots, custom_range, mesh_size):
    if neighborhood == 1:
        print('Add a relay next to a random connected relay\'s neighborhood.')
        return add_next_to_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)[:2]
    elif neighborhood == 2:
        print('Delete a random connected relay.')
        return delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [], custom_range)[:2]
    elif neighborhood == 3:
        print('Delete a random relay that\'s next to a sentinel with multiple relay neighbors.')
        return delete_relay_next_to_sentinel(sinked_sentinels, sinked_relays, free_slots, [], custom_range)[:2]
    elif neighborhood == 4:
        print('Swap relay with free slot.')
        return relocate_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)[:2]
    elif neighborhood == 5:
        print('Add a relay next to a sentinel with no relay neighbors.')
        return add_relay_next_to_sentinel(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)[:2]

def is_better_solution(grid, new_free_slots, sink, new_sinked_relays, sinked_sentinels, old_free_slots, old_sinked_relays, mesh_size, alpha, beta):
    new_fitness = epsilon_constraints(grid, new_free_slots, sink, new_sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
    old_fitness = epsilon_constraints(grid, old_free_slots, sink, old_sinked_relays, sinked_sentinels, 0, mesh_size, alpha, beta)
    
    if new_fitness < old_fitness:
        distance_bman, sentinel_bman, _ = dijkstra(grid, sink, new_sinked_relays, sinked_sentinels)
        return sentinel_bman.count(999) == 0
    return False

# Include the helper functions (add_next_to_relay, delete_random_relay, etc.) here...
'''
    5 Neighborhoods
'''

def add_next_to_relay(sentinels, sink, sinked_relays, free_slots, remember_used_relays, custom_range, mesh_size):
    performed_action = []
    candidate_slots = []
    
    allowed_sinked_relays = [x for x in sinked_relays if x not in remember_used_relays]
    min_meshes, max_meshes = get_min_max_meshes(allowed_sinked_relays, free_slots, custom_range)
    chosen_random_relay = max_meshes[0][0]
    
    for i in range(len(free_slots)):
        if math.dist(chosen_random_relay, free_slots[i]) < custom_range:
            candidate_slots.append(free_slots[i])
            
    if candidate_slots:
        chosen_random_slot = random.choice(candidate_slots)
        # sinked_relays.append((chosen_random_slot, (abs(sink[0] - chosen_random_slot[0]) + abs(sink[1] - chosen_random_slot[1])) / mesh_size))
        # sinked_relays.append((chosen_random_slot, 1))
        sinked_relays.append((chosen_random_slot, hop_count(sink, chosen_random_slot, sentinels, mesh_size)))
        performed_action = [1, chosen_random_slot, "(P) Add a relay next to a random connected relay's neighborhood."]
        free_slots.remove(chosen_random_slot)
        remember_used_relays.append(chosen_random_slot)
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def delete_random_relay(sentinels, sinked_relays, free_slots, remember_used_relays, custom_range):
    performed_action = []
    min_meshes, max_meshes = get_min_max_meshes(sinked_relays, free_slots, custom_range)
    chosen_random_relay = min_meshes[0][0]
    
    if sinked_relays:
        sinked_relays = [relay for relay in sinked_relays if relay[0] != chosen_random_relay]
        performed_action = [2, chosen_random_relay, "(P) Delete a random connected relay."]
        free_slots.append(chosen_random_relay)
        remember_used_relays.append(chosen_random_relay)
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def delete_relay_next_to_sentinel(sentinels, sinked_relays, free_slots, remember_used_relays, custom_range):
    performed_action = []
    
    # Dictionary to store the count of relays next to each sentinel
    sentinel_neighbor_count = {tuple(sentinel): 0 for sentinel in sentinels}
    
    # Count the number of relays next to each sentinel
    for sentinel in sentinels:
        for relay in sinked_relays:
            if math.dist(sentinel, relay[0]) < custom_range:
                sentinel_neighbor_count[tuple(sentinel)] += 1
    
    # Filter out sentinels with multiple relay neighbors
    sentinels_with_multiple_neighbors = [sentinel for sentinel, count in sentinel_neighbor_count.items() if count > 1]
    
    # Select a random sentinel with multiple relay neighbors, if any
    if sentinels_with_multiple_neighbors:
        chosen_sentinel = random.choice(sentinels_with_multiple_neighbors)
        
        # Find relays next to the chosen sentinel
        relays_next_to_chosen_sentinel = [relay for relay in sinked_relays if math.dist(chosen_sentinel, relay[0]) < custom_range]
        
        # Delete a random relay next to the chosen sentinel
        if relays_next_to_chosen_sentinel:
            chosen_random_relay = random.choice(relays_next_to_chosen_sentinel)
            sinked_relays = [relay for relay in sinked_relays if relay[0] != chosen_random_relay[0]]
            performed_action = [3, chosen_random_relay[0], "(P) Delete a random relay that's next to a sentinel with multiple relay neighbors"]
            free_slots.append(chosen_random_relay[0])
            remember_used_relays.append(chosen_random_relay[0])
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def relocate_relay(sinked_sentinels, sink, sinked_relays, free_slots, remember_used_relays, custom_range, mesh_size):
    performed_action = []
    
    if sinked_relays and free_slots:
        # Choose a relay and a free slot randomly
        '''relay_index = random.randint(0, len(sinked_relays) - 1)
        free_slot_index = random.randint(0, len(free_slots) - 1)
        
        # Swap the positions of the relay and the free slot
        relay_position = sinked_relays[relay_index][0]
        free_slot_position = free_slots[free_slot_index]
        sinked_relays[relay_index] = (free_slot_position, sinked_relays[relay_index][1])
        free_slots[free_slot_index] = relay_position
        sinked_relays = [relay for relay in sinked_relays if relay[0] != relay_position]

        # Update the performed action to reflect the swap
        performed_action = [2, "(LS) Swap relay with free slot"]'''
        free_slots, sinked_relays, action, remember_used_relays = delete_random_relay(sinked_sentinels, sinked_relays, free_slots, [], custom_range)
        free_slots, sinked_relays, action, remember_used_relays = add_next_to_relay(sinked_sentinels, sink, sinked_relays, free_slots, [], custom_range, mesh_size)
        # Update the performed action to reflect the swap
        performed_action = [2, "(LS) Swap relay with free slot"]
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def add_relay_next_to_sentinel(sentinels, sink, sinked_relays, free_slots, remember_used_relays, custom_range, mesh_size):
    performed_action = []
    
    for sentinel in sentinels:
        no_neighbors = True
        for relay in sinked_relays:
            if math.dist(sentinel, relay[0]) < custom_range:
                no_neighbors = False
                break
        
        if no_neighbors:
            candidate_slots = [slot for slot in free_slots if math.dist(sentinel, slot) < custom_range]
            if candidate_slots:
                chosen_slot = random.choice(candidate_slots)
                # sinked_relays.append((chosen_slot, (abs(sink[0] - chosen_slot[0]) + abs(sink[1] - chosen_slot[1])) / mesh_size))
                # sinked_relays.append((chosen_slot, 1))
                sinked_relays.append((chosen_slot, hop_count(sink, chosen_slot, sentinels, mesh_size)))
                performed_action = [1, chosen_slot, "(P) Add a relay next to a sentinel with no relay neighbors."]
                free_slots.remove(chosen_slot)
                remember_used_relays.append(chosen_slot)
                break  # Only add one relay per iteration
    
    return free_slots, sinked_relays, performed_action, remember_used_relays

def hop_count(sink, relay, sentinels, mesh_size):
    current_position = relay
    hop_count = 0
    # Iterate until the current position reaches the sink
    while current_position != sink:
        # Find neighbors of the current position
        neighbors = [(current_position[0] + 1, current_position[1]),
                     (current_position[0] - 1, current_position[1]),
                     (current_position[0], current_position[1] + 1),
                     (current_position[0], current_position[1] - 1)]
        neighbors = [neighbor for neighbor in neighbors if neighbor not in [position for position in sentinels]]
        # Calculate distances to each neighbor
        distances = [abs(sink[0] - neighbor[0]) + abs(sink[1] - neighbor[1]) for neighbor in neighbors]
        # Choose the neighbor with the minimum distance to the sink
        min_index = distances.index(min(distances))
        current_position = neighbors[min_index]
        hop_count += 1
    return int(hop_count /mesh_size)

def get_min_max_meshes(sinked_relays, free_slots, custom_range):
    min_meshes_candidate = []
    max_meshes_candidate = []
    random.shuffle(sinked_relays)

    for i in range(len(sinked_relays)):
        empty_meshes_counter = 0

        # Calculate meshes around a sinked relay
        for j in range(len(free_slots)):
            if math.dist(sinked_relays[i][0], free_slots[j]) < custom_range:
                empty_meshes_counter = empty_meshes_counter + 1

        if len(min_meshes_candidate) != 0 and len(max_meshes_candidate) != 0:

            # Acquire minimum meshes
            if min_meshes_candidate[1] > empty_meshes_counter:
                min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]

            # Acquire maximum meshes
            if max_meshes_candidate[1] < empty_meshes_counter:
                max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
        else:
            min_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
            max_meshes_candidate = [sinked_relays[i], empty_meshes_counter]
    return min_meshes_candidate, max_meshes_candidate