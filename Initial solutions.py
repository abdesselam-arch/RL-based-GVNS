from PersonalModules.Genetic import genetic_algorithm
from PersonalModules.utilities import display

def create(chosen_grid, sink_location, ):
    free_slots = []

    # Create a grid
    # chosen_grid = int(input("Choose your grid size: "))
    print("You chose the grid's size to be: ", chosen_grid, "*", chosen_grid)
    grid = chosen_grid * 20

    # Create a sink
    # sink_location = int(input("\nDo you want the sink in the middle of the grid? (Type 0 to choose a custom location) "))
    if sink_location == 0:
        sink_X_Axis = int(input("Enter the X coordinate of the sink."))
        sink_Y_Axis = int(input("Now enter the Y coordinate of the sink."))
        sink = (sink_X_Axis, sink_Y_Axis)
    else:
        if (chosen_grid % 2) == 0:
            sink = ((grid / 2) + 10, (grid / 2) - 10)
        elif (chosen_grid % 2) == 1:
            sink = (((grid - 20) / 2) + 10, ((grid - 20) / 2) + 10)

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

def main():

    grid, sink, sinkless_sentinels, free_slots = create(35, 1)
    max_hops_number = grid

    print("\n   Fixed Initial solution given by genetic")
    # genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots, Finished, ERROR = genetic_algorithm(2, 0, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range = 30, mesh_size = 20)
    
    with open("Initial solutions/genetic_sinked_sentinels_35.txt", "r") as f:
        genetic_sinked_sentinels = eval(f.read())
    with open("Initial solutions/genetic_sinked_relays_35.txt", "r") as f:
        genetic_sinked_relays = eval(f.read())
    genetic_free_slots = []
    
    print(f'genetic_sinked_sentinels = {genetic_sinked_sentinels}')
    print(f'genetic_sinked_relays = {genetic_sinked_relays}')
    print(f'genetic_free_slots = {genetic_free_slots}')

    display(grid, sink, genetic_sinked_relays, genetic_sinked_sentinels, title="Genetic Algorithm")

    '''with open("genetic_sinked_sentinels_35.txt", "w") as f:
        f.write(str(genetic_sinked_sentinels))
    with open("genetic_sinked_relays_35.txt", "w") as f:
        f.write(str(genetic_sinked_relays))'''

if __name__ == '__main__':
    main()