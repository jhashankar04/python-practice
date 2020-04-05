from operator import itemgetter
def get_next_best_step(paths_so_far, gns, hns, fns): 
    min_fns = min(fns)
    idx  = [i for i,val in enumerate(fns) if val==min_fns]

    if len(idx) > 1: 
        dests = [paths_so_far[i][-1] for i in idx]
        selected_idx = idx[dests.index(min(dests))]
    else:
        selected_idx = idx[0]
    path = paths_so_far[selected_idx]
    gn   = gns[selected_idx]
    
    # print("All Paths:")
    # print(paths_so_far)
    # print("hn")
    # print(hns)
    # print("Selected path")
    # print(path)
    
    #paths_so_far.pop(selected_idx)
    #gns.pop(selected_idx)
    #hns.pop(selected_idx)
    #fns.pop(selected_idx)
    
    #return ({'best_path': path, 'best_gn': gn, 'paths_so_far': paths_so_far, 
    #         'gns':gns, 'hns':hns, 'fns':fns})
    return selected_idx

no_of_nodes = int(input()) 

heuristics  = list(map(int, input().split(" ")))

start_end_node = list(map(int, input().split(" ")))
start_node = start_end_node[0]
end_node = start_end_node[1]

no_of_edges = int(input()) 

sources = []
destinations = []
costs =[]

for i in range(no_of_edges):
    row = list(map(int, input().split(" ")))
    sources.append(row[0])
    destinations.append(row[1])
    costs.append(row[2])
    
edges = {'source':sources, 'destination':destinations, 'cost':costs}

#print(edges)
#print(heuristics)

current_node = start_node

paths_so_far  = []
gns           = []
hns           = []
fns           = []
destination_nodes = []
current_path = []
cost_so_far  = 0
nodes_explored = [start_node]

while current_node != end_node:
    
    # Explore new paths from current node
    possible_destinations_indices = [i for i,val in enumerate(edges['source']) if val==current_node]
    no_of_destinations = len(possible_destinations_indices)
    
    if current_node == start_node:
        current_path = [start_node]
        cost_so_far  = 0
       
    for i in range(no_of_destinations):
        # if len(current_path) >=2 : 
        #     if current_path[-2] == edges['destination'][possible_destinations_indices[i]]: 
        #         continue;
        gn = edges['cost'][possible_destinations_indices[i]] 
        gns.append(cost_so_far + gn)
        
        this_path = current_path.copy()
        this_path.append(edges['destination'][possible_destinations_indices[i]])
        paths_so_far.append(this_path)
        
        hns.append(heuristics[edges['destination'][possible_destinations_indices[i]]-1])
        
        fns.append(gns[-1] + hns[-1])
    
    #best_step = get_next_best_step(paths_so_far, gns, hns, fns)
    #current_path = best_step['best_path']
    #cost_so_far  = best_step['best_gn']
    #current_node = current_path[-1]
    index_to_remove = get_next_best_step(paths_so_far, gns, hns, fns)
    current_path = paths_so_far[index_to_remove]
    cost_so_far = gns[index_to_remove]
    #index_to_remove = paths_so_far.pop(current_path)
    paths_so_far.pop(index_to_remove)
    gns.pop(index_to_remove)
    hns.pop(index_to_remove)
    fns.pop(index_to_remove)
    current_node = current_path[-1]
    
    #paths_so_far = best_step['paths_so_far']
    #gns = best_step['gns']
    #hns = best_step['hns']
    #fns = best_step['fns']
    
    #if current_node not in nodes_explored:
    nodes_explored.append(current_node)
    
    # print(paths_so_far)
    # print(gns)
    # print(hns)

#print(" ".join(list(map(str, current_path))))
print(" ".join(list(map(str, nodes_explored))))
print(str(cost_so_far))