import json
import math
import time
import os

from PersonalModules.Genetic import genetic_algorithm
from PersonalModules.UCB_VND import UCB_VND
from PersonalModules.generalVNS import GVNS
from PersonalModules.utilities import bellman_ford, dijkstra, display, get_Diameter, get_stat, len_sinked_relays


def create(chosen_grid, sink_location):
    free_slots = []

    # Create a grid
    print("You chose the grid's size to be: ", chosen_grid, "*", chosen_grid)
    grid = chosen_grid * 20
    print("Grid size: ", grid)
    print("Sink location: ", sink_location)

    # Create a sink
    if sink_location == 1:  # center
        if (chosen_grid % 2) == 0:
            sink = ((grid / 2) + 10, (grid / 2) - 10)
        elif (chosen_grid % 2) == 1:
            sink = (((grid - 20) / 2) + 10, ((grid - 20) / 2) + 10)
    elif sink_location == 2:  # top left
        sink = (50, grid - 50)
        print("Sink location: ", sink)
    elif sink_location == 3:  # top right
        sink = (grid - 50, grid - 50)
        print("Sink location: ", sink)
    elif sink_location == 4:  # bottom left
        sink = (50, 50)
        print("Sink location: ", sink)
    elif sink_location == 5:  # bottom right
        sink = (grid - 50, 50)
        print("Sink location: ", sink)
    else:
        # Default sink location if sink_location is invalid
        print("Invalid sink location. Defaulting to center.")
        sink = ((grid / 2) + 10, (grid / 2) - 10)

    # Create sentinels
    sinkless_sentinels = [(x, 10) for x in range(10, grid + 10, 20)] + \
                         [(x, grid - 10) for x in range(10, grid + 10, 20)] + \
                         [(10, y) for y in range(30, grid - 10, 20)] + \
                         [(grid - 10, y) for y in range(30, grid - 10, 20)]

    # Create the free slots
    for x in range(30, grid - 10, 20):
        for y in range(30, grid - 10, 20):
            if sink != (x, y):
                free_slots.append((x, y))
    return grid, sink, sinkless_sentinels, free_slots

def get_ordinal_number(n):
    if n % 100 in [11, 12, 13]:
        suffix = "th"
    else:
        last_digit = n % 10
        if last_digit == 1:
            suffix = "st"
        elif last_digit == 2:
            suffix = "nd"
        elif last_digit == 3:
            suffix = "rd"
        else:
            suffix = "th"
    return str(n) + suffix

'''def isPrime(x):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    tf = eng.isprime(x)
    return tf'''

def initial_solution(grid, sink, sinkless_sentinels, free_slots, max_hops_number, sink_location):
    genetic_free_slots = []

    print("You chose the grid's size to be: ", grid, "*", grid)
    print("You chose the sink's location to be: ", sink_location)
    folder_path = f"Initial solutions {sink_location}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    sentinel_file = os.path.join(folder_path, f"genetic_sinked_sentinels_{int(grid)}.txt")
    relay_file = os.path.join(folder_path, f"genetic_sinked_relays_{int(grid)}.txt")
    stats_file = os.path.join(folder_path, f"genetic_statistics_{int(grid)}.json")

    # Check if all three files exist
    if os.path.exists(sentinel_file) and os.path.exists(relay_file) and os.path.exists(stats_file):
        with open(sentinel_file, "r") as f:
            genetic_sinked_sentinels = eval(f.read())
        with open(relay_file, "r") as f:
            genetic_sinked_relays = eval(f.read())
        with open(stats_file, "r") as f:
            stats = json.load(f)
        print("Loaded existing initial solution and statistics.")
    else:
        # If files do not exist, generate new initial solution
        start_time = time.time()

        genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots, Finished, ERROR = genetic_algorithm(
            15, 10, sink, sinkless_sentinels, free_slots, max_hops_number + 1,
            custom_range=30, mesh_size=20
        )

        total_seconds = int(time.time() - start_time)

        # Convert seconds to H:M:S
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{seconds:02.0f}S"

        # Save the sentinel and relay lists
        with open(sentinel_file, "w") as f:
            f.write(str(genetic_sinked_sentinels))
        with open(relay_file, "w") as f:
            f.write(str(genetic_sinked_relays))

        # Save the statistics as JSON
        stats = {
            "number_of_sentinels": len(genetic_sinked_sentinels),
            "number_of_relays": len(genetic_sinked_relays),
            "grid_size": int(grid),
            "execution_time": time_string,
        }
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=4)

        print("New initial solution generated and statistics saved.")

    print(f"Initial Solution Stats -> Sentinels: {stats['number_of_sentinels']}, Relays: {stats['number_of_relays']}, Execution Time: {stats['execution_time']}")
    return genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots

def main():
    get_in = True
    # Create everything
    if get_in:
        
        # If needed to change the grid size or sink location, change the parameters here
        grid_size = 11
        sink_location = 1  # 1: center, 2: top left, 3: top right, 4: bottom left, 5: bottom right
        
        grid, sink, sinkless_sentinels, free_slots = create(grid_size, sink_location)
        max_hops_number = grid

    #user_input = int(input("     Type 1 for multiple times VNS.\n"))
    user_input = 1

    if user_input == 1:
        executions = 1
        vns_avg_hops = 0
        vns_avg_relays = 0
        vns_avg_performance = 0
        vns_avg_diameter = 0
        ga_avg_hops = 0
        ga_avg_relays = 0
        ga_avg_performance = 0
        ga_avg_diameter = 0

        #print("You chose Multiple times Greedy !\n")
        #user_input = int(input("How many Greedy executions you want to perform?"))
        # Change the user_input value to change the number of simulations (executions)
        user_input = 1

        '''
        Change the value of Gvns_or_RLGVNS to change the algorithm to execute
        1: executes GVNS
        any other value: executes RL-based GVNS (UCB_GVNS)
        '''
        Gvns_or_RLGVNS = 2

        simulation_start_time = time.time()
        execution_times = []
        while executions <= user_input:
            print("\n # This is the ", get_ordinal_number(executions), " execution.")

            start_time = time.time()
            # Generate the initial solution using greedy algorithm
            print("\nFixed Initial solution given by genetic ")
            genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots = initial_solution(grid/20, sink, sinkless_sentinels, free_slots, max_hops_number+1, sink_location)
            print("---------------------------------------")

            # Get the performance before VNS, perform VNS then Get the performance after VNS
            print("\n   Please wait until some calculations are finished...")
            #distance_bman, sentinel_bman, genetic_cal_bman = bellman_ford(grid, genetic_free_slots, sink, genetic_sinked_relays,
            #                                                        genetic_sinked_sentinels)
            distance_bman, sentinel_bman, genetic_cal_bman = dijkstra(grid, sink, genetic_sinked_relays, genetic_sinked_sentinels)

            performance_before, relays_before, hops_before = get_stat(genetic_sinked_relays, sentinel_bman, genetic_cal_bman, grid, genetic_free_slots, sink, genetic_sinked_sentinels, mesh_size = 20, alpha = 0.5, beta = 0.5, gen_diameter=int(grid/20))
            diameter_before = get_Diameter(sentinel_bman, genetic_cal_bman, mesh_size = 20)
            print("   Calculations are done !")

            sinked_relays = genetic_sinked_relays
            sinked_sentinels = genetic_sinked_sentinels
            free_slots = genetic_free_slots

            ga_diameter = diameter_before
            ga_avg_hops += hops_before
            ga_avg_relays += relays_before
            ga_avg_performance += performance_before
            ga_avg_diameter += ga_diameter

            # display(grid, sink, sinked_relays, sinked_sentinels, title="Genetic Algorithm")
            print('Starting the main algorithm now!!')

            if Gvns_or_RLGVNS == 1:
                sinked_relays, free_slots = GVNS(grid, sink, sinked_sentinels, sinked_relays, free_slots, 30, 20, max_iterations=1, alpha=0.5, beta=0.5, gen_diameter=ga_diameter)
                print("   General Variable Neighborhood Search algorithm finished execution successfully !")
            else:    
                sinked_relays, free_slots = UCB_VND(grid, sink, sinked_sentinels, sinked_relays, free_slots, 30, 20, lmax=5, alpha=0.5, beta=0.5, gen_diameter=ga_diameter)
                print("   Upper Confidence Bounde + General Variable Neighborhood Search algorithm finished execution successfully !")

            print("\n   Please wait until some calculations are finished...")
            #distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
            
            distance_bman, sentinel_bman, cal_bman = dijkstra(grid, sink, sinked_relays, sinked_sentinels)
            performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size = 20, alpha = 0.5, beta = 0.5, gen_diameter=ga_diameter)
            
            diameter_after = get_Diameter(sentinel_bman, cal_bman, mesh_size = 20)
            relays_after = len_sinked_relays(sinked_relays)

            # display(grid, sink, sinked_relays, sinked_sentinels, title="UCB VND Algorithm")
            print("   Calculations are done !")

            print(f"\nFitness BEFORE: {performance_before}")
            print(f"Fitness AFTER: {performance_after}\n")
                
            print(f"Relays BEFORE: {relays_before}")
            print(f"Relays AFTER: {relays_after}\n")

            print(f"Network diameter BEFORE: {diameter_before}")
            print(f"Network diameter AFTER: {diameter_after}\n")

            print(f"Hops Average BEFORE: {hops_before}")
            print(f"Hops Average AFTER: {hops_after}\n")

            vns_avg_hops += hops_after
            vns_avg_relays += relays_after
            vns_avg_performance += performance_after
            vns_avg_diameter += diameter_after

            executions = executions + 1

            end_time = time.time()
            # GET TIME
            total_time = int(end_time - start_time)
            execution_times.append(total_time)
            hours, remainder = divmod(total_time, 3600)
            minutes, remainder = divmod(remainder, 60)
            time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

            stats_file = os.path.join(f"Initial solutions {sink_location}", f"genetic_statistics_{int(grid/20)}.json")
            with open(stats_file, "r") as f:
                stats = json.load(f)
            GA_time_string = stats["execution_time"]
            print(f'Execution time: {time_string}')

        simulation_end_time = time.time()
        # GET TIME
        total_time = int(simulation_end_time - simulation_start_time)
        hours, remainder = divmod(total_time, 3600)
        minutes, remainder = divmod(remainder, 60)
        simulation_time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

        total_execution_time = sum(execution_times)
        #   Calculate average execution time
        average_execution_time = total_execution_time / len(execution_times)
        hours, remainder = divmod(average_execution_time, 3600)
        minutes, remainder = divmod(remainder, 60)
        avg_time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

        print('\n\nSimulation Results:\n')

        print('Initial GA Results AVERAGE:')

        print(f'Relays AVERAGE: {math.ceil(ga_avg_relays / user_input)}')
        print(f'Hops AVERAGE: {math.ceil(ga_avg_hops / user_input)}')
        print(f'Performance AVERAGE: {ga_avg_performance / user_input}')
        print(f'Diameter AVERAGE: {math.ceil(ga_avg_diameter / user_input)}')
        
        if Gvns_or_RLGVNS == 1:
            print('\nGVNS Results AVERAGE:')
        else:
            print('\nUCB_GVNS Results AVERAGE:')

        print(f'Relays AVERAGE: {math.ceil(vns_avg_relays / user_input)}')
        print(f'Hops AVERAGE: {math.ceil(vns_avg_hops / user_input)}')
        print(f'Performance AVERAGE: {vns_avg_performance / user_input}')
        print(f'Diameter AVERAGE: {math.ceil(vns_avg_diameter / user_input)}')

        avg_execution_time = total_time / user_input
        avg_hours, avg_remainder = divmod(avg_execution_time, 3600)
        avg_minutes, avg_remainder = divmod(avg_remainder, 60)
        avg_time_string = f"{avg_hours:02.0f}H_{avg_minutes:02.0f}M_{avg_remainder:02.0f}       "

        print(f'\nGenetic Execution time: {GA_time_string}')
        print(f'\nExecution time AVERAGE: {avg_time_string}')
        print(f'Total execution time: {time_string}')
   
if __name__ == '__main__':
    main()